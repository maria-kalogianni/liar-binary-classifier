"""
explainer.py — Explainability pipeline for TruthClassifier predictions.

Explanation strategy (tested in day7_explainability.ipynb):

    Step 1 — LayerIntegratedGradients (Captum): which WORDS drove the prediction?
             Freezes CS + metadata, varies text via interpolation.
             Completeness check: if delta < 0.20 → use IG; else → LIME fallback.

    Step 2 — GradientSHAP (Captum): which NUMERICAL features (CS + metadata) mattered?
             Freezes text, varies numerical features.

    Step 3 — Claude API: converts raw attribution scores into plain-text explanation.
             Rules: no external knowledge, no Markdown, 2-3 sentences, general audience.

Thread safety: all Captum "frozen" state is stored as instance attributes (self._lig_cs,
self._gs_input_ids, etc.) rather than module-level globals, so multiple Explainer
instances cannot interfere with each other.
"""

import os
import numpy as np
import torch

# Names of the 5 numerical features passed to GradientSHAP (in order)
# Order must match how we build num_input in _explain_numerical():
#   col 0: cs_ratio, cols 1-4: [speaker_cred, affil_cred, ctx_cred, len_norm]
FEATURE_NAMES = ['cs_ratio', 'speaker_cred', 'affil_cred', 'ctx_cred', 'len_norm']


class Explainer:
    """
    Generates human-readable explanations for TruthClassifier predictions.

    Usage:
        explainer = Explainer(model, tokenizer, artifacts, device)
        explanation = explainer.explain(prep, prob, label)

    Args (constructor):
        model      : TruthClassifier instance in eval() mode
        tokenizer  : DistilBertTokenizerFast (loaded from artifacts)
        artifacts  : dict from preprocessors.pkl
        device     : torch.device (cpu or cuda)
    """

    def __init__(self, model, tokenizer, artifacts, device):
        # ── Lazy imports — only when Explainer is actually instantiated ────────
        try:
            from captum.attr import LayerIntegratedGradients, GradientShap as _GS
        except ImportError:
            raise ImportError(
                'captum is required for explainability.\n'
                'Install with: pip install captum'
            )
        try:
            import anthropic as _anthropic
            _anthropic_available = True
        except ImportError:
            # anthropic not installed -> use rule-based fallback
            _anthropic_available = False
            _anthropic = None

        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device

        # Values from artifacts needed for explanation logic
        self.default_cred   = artifacts.get('default_cred', 0.557)
        self.best_threshold = artifacts.get('best_threshold', 0.57)
        self.model_config   = artifacts.get('model_config', {})
        # cs_default is hardcoded — it is NOT stored in preprocessors.pkl
        self.cs_default     = 0.5

        # ── Captum "frozen" state ─────────────────────────────────────────────
        # These are set immediately before each Captum call and hold the values
        # that should NOT change during gradient interpolation/estimation.
        #
        # LayerIG varies input_ids (text) → CS and metadata must be frozen:
        self._lig_cs   = None   # (1, 1) tensor — set in _explain_text()
        self._lig_meta = None   # (1, 4) tensor — set in _explain_text()
        #
        # GradientSHAP varies numerical features → text must be frozen:
        self._gs_input_ids  = None   # (1, max_len) — set in _explain_numerical()
        self._gs_attn_mask  = None   # (1, max_len) — set in _explain_numerical()

        # ── Anthropic client ──────────────────────────────────────────────────
        api_key = os.environ.get('ANTHROPIC_API_KEY', '')
        if not _anthropic_available:
            # anthropic package not installed
            print('[Explainer] NOTE: anthropic not installed. '
                  'Using rule-based explanation fallback.')
            self._claude_client = None
        elif not api_key:
            # package installed but no API key
            print('[Explainer] NOTE: ANTHROPIC_API_KEY not set. '
                  'Using rule-based explanation fallback.')
            self._claude_client = None
        else:
            self._claude_client = _anthropic.Anthropic(api_key=api_key)

        # ── Forward wrappers for Captum ───────────────────────────────────────
        # Captum calls these functions directly with positional arguments —
        # we cannot use class methods (which receive `self` as first arg).
        # We use CLOSURES instead: inner functions that capture `self` from
        # the enclosing __init__ scope. This is equivalent to using globals
        # but is thread-safe because each Explainer instance has its own closure.

        def _forward_for_lig(input_ids, attention_mask):
            """
            Wrapper called by LayerIG during gradient interpolation.
            Called many times with interpolated input_ids (from baseline to actual).
            CS and metadata stay frozen via self._lig_cs / self._lig_meta.

            Shape note: Captum may call this with batch_size > 1 (interpolation steps
            are batched internally). We use .expand() to match the current batch size
            without copying memory.
            """
            batch_size = input_ids.shape[0]
            # expand: (1, 1) → (batch, 1), (1, 4) → (batch, 4)
            # expand creates a VIEW — no memory copy, very fast
            cs   = self._lig_cs.expand(batch_size, -1)
            meta = self._lig_meta.expand(batch_size, -1)
            out  = self.model(input_ids, attention_mask, cs, meta)
            return out.squeeze(-1)   # (batch, 1) → (batch,)  ← Captum requires 1D output

        def _forward_for_gs(numerical_features):
            """
            Wrapper called by GradientSHAP during stochastic gradient estimation.
            numerical_features: (batch, 5) — [cs_ratio, spk_cred, afl_cred, ctx_cred, len_norm]
            Text inputs stay frozen via self._gs_input_ids / self._gs_attn_mask.

            CRITICAL: .float() cast is REQUIRED.
            GradientSHAP internally creates float64 (Double) tensors during estimation,
            but our model weights are float32. Without this cast, PyTorch raises:
              RuntimeError: expected scalar type Float but found Double
            """
            numerical_features = numerical_features.float()   # float64 → float32

            batch_size = numerical_features.shape[0]
            cs   = numerical_features[:, 0:1]   # first column → (batch, 1)
            meta = numerical_features[:, 1:5]   # columns 1-4  → (batch, 4)

            ids  = self._gs_input_ids.expand(batch_size, -1)
            mask = self._gs_attn_mask.expand(batch_size, -1)

            out = self.model(ids, mask, cs, meta)
            return out.squeeze(-1)   # (batch,)

        # Attach LayerIG to the word embedding layer of DistilBERT.
        # The word embedding layer converts integer token IDs → 768-dim vectors.
        # LayerIG attributes the output score to changes in these embedding vectors.
        self.lig = LayerIntegratedGradients(
            _forward_for_lig,
            model.distilbert.embeddings.word_embeddings,
        )

        # GradientSHAP attributes the output score to each of the 5 numerical features.
        # _GS is GradientShap imported at the top of __init__ (lazy import)
        self.gs = _GS(_forward_for_gs)

    # ── Helper: merge WordPiece subword tokens ────────────────────────────────

    def _merge_subword_tokens(self, token_score_list):
        """
        DistilBERT uses WordPiece tokenisation, which splits words into subwords:
            'unemployment' → ['un', '##employ', '##ment']

        Tokens starting with '##' are continuations of the previous token.
        This function merges them back into full words and SUMS their scores.

        Args:
            token_score_list : list of (token_string, float_score)

        Returns:
            list of (merged_word_string, summed_score)

        Example:
            [('un', 0.01), ('##employ', 0.03), ('##ment', 0.02), ('is', 0.05)]
            →  [('unemployment', 0.06), ('is', 0.05)]
        """
        merged = []
        for token, score in token_score_list:
            if token.startswith('##') and merged:
                # Continuation token: attach to previous word, sum the score
                prev_word, prev_score = merged[-1]
                merged[-1] = (prev_word + token[2:], prev_score + score)
            else:
                # New word starts here
                merged.append((token, score))
        return merged

    # ── Text attribution: LayerIntegratedGradients ────────────────────────────

    def _explain_text(self, prep, n_steps=20):
        """
        Run LayerIG to find which words contributed most to the prediction.

        The method computes a Riemann sum integral of gradients from a baseline
        (all [PAD] tokens) to the actual input. More steps = better approximation.

        n_steps=20 is appropriate for CPU. Use 50+ on GPU for tighter bounds.

        Args:
            prep    : dict from _prepare_input() in predict.py
            n_steps : number of interpolation steps (trade-off: quality vs speed)

        Returns:
            token_scores : list of (word_string, float_score)
                           positive score → word pushes toward TRUE
                           negative score → word pushes toward FALSE
            delta        : convergence quality measure (lower is better)
                           < 0.05  → excellent
                           < 0.20  → acceptable (normal for 20-50 steps on CPU)
                           ≥ 0.20  → poor → trigger LIME fallback
        """
        # Freeze CS and metadata for this run (text is the only varying input)
        self._lig_cs   = prep['cs_tensor'].detach()
        self._lig_meta = prep['meta_tensor'].detach()

        input_ids      = prep['input_ids']       # (1, max_len)
        attention_mask = prep['attention_mask']  # (1, max_len)

        # Baseline: all tokens are [PAD] (token ID = 0 in DistilBERT).
        # Represents "no information" — the starting point of the integral.
        baseline_ids = torch.zeros_like(input_ids)

        # IMPORTANT: model.eval() MUST be called before LayerIG.
        # In training mode, Dropout adds randomness → IG scores become non-deterministic.
        # In eval mode, Dropout is disabled → scores are reproducible.
        self.model.eval()

        print(f'  LayerIG ({n_steps} steps)...', end=' ', flush=True)

        # lig.attribute() computes integrated gradients:
        # - inputs        : the actual token IDs we want to explain
        # - baselines     : the reference point (all [PAD])
        # - additional_forward_args: attention_mask is passed as-is (not varied)
        # - return_convergence_delta: also returns a quality metric
        attributions, delta = self.lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            n_steps=n_steps,
            return_convergence_delta=True,
        )
        print('done.')

        # attributions shape: (1, max_len, 768)
        # We sum across the 768 embedding dimensions to get one score per token
        # .sum(dim=-1): (1, max_len, 768) → (1, max_len)
        # .squeeze(0):  (1, max_len) → (max_len,)
        token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()

        # Convert integer token IDs back to readable strings (e.g. 2009 → 'the')
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

        # Keep only real tokens (attention_mask=1), skip [PAD] positions
        mask = attention_mask[0].cpu().tolist()
        filtered = [
            (tok, float(score))
            for tok, score, m in zip(tokens, token_scores, mask)
            if m == 1   # m=1 → real token; m=0 → padding
        ]

        # Merge '##subword' tokens back into full words
        merged = self._merge_subword_tokens(filtered)

        # delta is a tensor — .item() converts to Python float, abs() for magnitude
        delta_val = abs(delta.item())
        return merged, delta_val

    # ── Text attribution fallback: LIME ───────────────────────────────────────

    def _explain_text_lime(self, prep, num_samples=50):
        """
        LIME fallback for text attribution (used when LayerIG delta ≥ 0.20).

        LIME works differently from IG:
        - It randomly removes words from the statement
        - It runs the model on each perturbed version
        - It fits a simple linear model to approximate which words matter

        LIME is model-agnostic (treats the model as a black box) and always converges,
        but it's less mathematically faithful than LayerIG.

        CS and metadata are FROZEN (same speaker for all perturbed texts).

        Args:
            prep        : dict from _prepare_input() in predict.py
            num_samples : number of perturbed texts to generate (50 is fast on CPU)

        Returns:
            list of (word_string, float_score) — positive = toward TRUE
        """
        from lime.lime_text import LimeTextExplainer

        max_len = self.model_config.get('max_len', 128)

        def _lime_predict_fn(texts):
            """
            Called by LIME with a list of perturbed text versions.
            Returns array of shape (n_texts, 2): [[P(FALSE), P(TRUE)], ...]
            LIME needs probabilities for BOTH classes (not just TRUE).
            """
            results = []
            for text in texts:
                enc = self.tokenizer(
                    text,
                    max_length=max_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                )
                with torch.no_grad():
                    logit = self.model(
                        enc['input_ids'].to(self.device),
                        enc['attention_mask'].to(self.device),
                        prep['cs_tensor'],    # FROZEN: same speaker for all texts
                        prep['meta_tensor'],  # FROZEN: same metadata
                    )
                p_true = torch.sigmoid(logit).item()
                results.append([1 - p_true, p_true])   # [P(FALSE), P(TRUE)]
            return np.array(results)

        lime_explainer = LimeTextExplainer(class_names=['FALSE', 'TRUE'])

        lime_exp = lime_explainer.explain_instance(
            prep['statement'],
            _lime_predict_fn,
            num_features=10,    # return top 10 words
            num_samples=num_samples,
        )

        return lime_exp.as_list()   # list of (word, score)

    # ── Numerical attribution: GradientSHAP ───────────────────────────────────

    def _explain_numerical(self, prep, n_samples=50):
        """
        Run GradientSHAP to find which numerical features mattered most.

        The 5 features are: [cs_ratio, speaker_cred, affil_cred, ctx_cred, len_norm]
        (cs_ratio and metadata combined into one 5-dim input).

        GradientSHAP differs from plain IG: it requires a DISTRIBUTION of baselines
        and adds noise during gradient estimation (stochastic approximation).
        This gives more stable estimates than a single baseline.

        Text is FROZEN during this computation.

        Args:
            prep      : dict from _prepare_input() in predict.py
            n_samples : number of gradient samples (50 is appropriate for CPU)

        Returns:
            dict: {feature_name: attribution_score}
            positive → pushes toward TRUE, negative → pushes toward FALSE
        """
        # Freeze text inputs for this run
        self._gs_input_ids = prep['input_ids'].detach()
        self._gs_attn_mask = prep['attention_mask'].detach()

        # Combine cs_ratio (1,) and metadata (4,) into one (1, 5) tensor
        num_input = torch.cat(
            [prep['cs_tensor'], prep['meta_tensor']], dim=1
        ).to(self.device)   # shape: (1, 5)

        # Baseline distribution: 50 samples near the "neutral" default values.
        # GradientSHAP requires a distribution (not a single baseline like IG).
        # Neutral = [cs_default, default_cred, default_cred, default_cred, 0.0]
        # Meaning: unknown speaker (CS=0.5), average credibility, average length.
        torch.manual_seed(42)   # fixed seed → reproducible baselines
        neutral = torch.tensor(
            [[self.cs_default, self.default_cred, self.default_cred,
              self.default_cred, 0.0]]   # shape: (1, 5)
        )
        # Add small Gaussian noise to create a distribution of 50 baselines
        # * 0.05: small perturbations so baselines stay near the neutral point
        noise         = torch.randn(50, 5) * 0.05
        baseline_dist = (neutral + noise).to(self.device)   # shape: (50, 5)

        print(f'  GradientSHAP ({n_samples} samples)...', end=' ', flush=True)

        # gs.attribute() returns shape (1, 5) — one attribution per feature
        # stdevs=0.09: std of additional Gaussian noise during gradient estimation
        attrs = self.gs.attribute(
            inputs=num_input,
            baselines=baseline_dist,
            n_samples=n_samples,
            stdevs=0.09,
        )
        print('done.')

        # squeeze(0): (1, 5) → (5,)
        # .tolist(): numpy array → Python list of 5 floats
        attrs_list = attrs.squeeze(0).detach().cpu().numpy().tolist()

        # Zip feature names with attribution values → dict
        return dict(zip(FEATURE_NAMES, attrs_list))

    # ── Claude API prompt builder ─────────────────────────────────────────────

    def _build_prompt(self, prep, prob, label, text_attrs, num_attrs, use_ig):
        """
        Build the prompt for Claude API from attribution scores.

        Prompt design principles (CLAUDE.md):
        - Plain text only, no Markdown
        - ONLY use evidence provided — no external knowledge about the claim
        - Skip cs_ratio if speaker is unknown (cs_known=False)
        - Skip speaker_cred if it is near the default (unknown speaker)
        - 2-3 sentences, general audience

        Args:
            prep       : dict from _prepare_input()
            prob       : P(TRUE) float from model
            label      : 'TRUE' or 'FALSE'
            text_attrs : list of (word, score) from LayerIG or LIME
            num_attrs  : dict {feature: score} from GradientSHAP
            use_ig     : True if text_attrs came from LayerIG (else LIME)

        Returns:
            prompt string
        """
        # Confidence = probability of the PREDICTED class (not always P(TRUE))
        confidence_pct = round(prob * 100) if label == 'TRUE' else round((1 - prob) * 100)

        # ── Top 3 words from text attribution ─────────────────────────────────
        SKIP_TOKENS = {'[CLS]', '[SEP]', '[PAD]'}

        if use_ig:
            # LayerIG may include special tokens — filter them out
            sorted_text = sorted(
                [(t, s) for t, s in text_attrs if t not in SKIP_TOKENS],
                key=lambda x: abs(x[1]),    # sort by absolute attribution value
                reverse=True,               # largest first
            )
        else:
            # LIME returns word-level results — no special tokens to filter
            sorted_text = sorted(text_attrs, key=lambda x: abs(x[1]), reverse=True)

        top_words = [tok for tok, _ in sorted_text[:3]]

        # ── Numerical evidence lines (max 3) ──────────────────────────────────
        # Sort by absolute attribution so the most impactful features come first
        sorted_num = sorted(num_attrs.items(), key=lambda x: abs(x[1]), reverse=True)
        num_lines  = []

        for feat, score in sorted_num:
            if len(num_lines) >= 3:
                break   # stop after 3 lines

            direction = 'supports TRUE' if score > 0 else 'supports FALSE'

            if feat == 'cs_ratio':
                # Skip cs_ratio if the speaker was NOT found in training data.
                # An unknown speaker gets cs_ratio = cs_default = 0.5 (neutral).
                # The model cannot distinguish them from truly neutral speakers,
                # so the attribution ≈ 0 and mentioning it would be misleading.
                if not prep.get('cs_known', True):
                    continue
                val  = prep['cs_val']
                line = (
                    f"the speaker's falsehood score is {val:.2f}/1.0 ({direction})"
                )

            elif feat == 'speaker_cred':
                val = prep['spk_cred']
                # Skip if speaker credibility is very close to the default value.
                # This means the speaker was not in training data and we used the
                # overall average — not meaningful to mention to the user.
                if abs(val - self.default_cred) < 0.02:
                    continue
                line = (
                    f"this speaker has been truthful {val*100:.0f}% of the time "
                    f"historically ({direction})"
                )

            elif feat == 'affil_cred':
                val  = prep['afl_cred']
                line = (
                    f"this speaker's political affiliation has a "
                    f"{val*100:.0f}% truthfulness rate ({direction})"
                )

            elif feat == 'ctx_cred':
                val  = prep['ctx_cred']
                line = (
                    f"statements made in this context are true "
                    f"{val*100:.0f}% of the time ({direction})"
                )

            elif feat == 'len_norm':
                len_val = prep['len_norm']
                if len_val > 1.0:
                    length_desc = 'unusually long'
                elif len_val < -1.0:
                    length_desc = 'unusually short'
                else:
                    length_desc = 'typical length'
                line = f"the statement is {length_desc} ({direction})"

            else:
                # Fallback for any unexpected feature name
                line = f"{feat} = {score:+.3f} ({direction})"

            num_lines.append(f'   - {line}')

        # ── Assemble final prompt ─────────────────────────────────────────────
        prompt = (
            f"You are explaining a political fact-checking decision. "
            f"You ONLY have access to the evidence listed below. "
            f"Do NOT use any external knowledge or verify the claim yourself.\n\n"
            f"Statement: \"{prep['statement']}\"\n"
            f"Classification: {label} (confidence {confidence_pct}%)\n\n"
            f"Evidence used:\n"
            f"1. Key words that influenced the decision: {', '.join(top_words)}\n"
            f"2. Speaker/context signals:\n"
            + '\n'.join(num_lines)
            + "\n\n"
            f"Write a 2-3 sentence explanation for a general audience.\n"
            f"Strict rules:\n"
            f"- Plain text only. No Markdown, no headers, no bold, no bullet points.\n"
            f"- Do NOT verify the statement or use any external knowledge.\n"
            f"- ONLY explain what the evidence above suggests about why it is likely {label}.\n"
            f"- No technical terms (no 'model', 'AI', 'algorithm', 'attribution').\n"
            f"- Be direct but acknowledge uncertainty where appropriate."
        )

        # Store intermediate results so _rule_based_explanation() can use them
        # if Claude API is not available. These are set here because _build_prompt
        # already has top_words and num_lines fully computed.
        self._last_label      = label
        self._last_confidence = confidence_pct
        self._last_top_words  = top_words    # list of word strings
        self._last_num_lines  = num_lines    # list of '   - ...' evidence strings

        return prompt

    # ── Claude API call ───────────────────────────────────────────────────────

    def _call_claude(self, prompt):
        """
        Send the prompt to Claude Haiku and return the explanation text.

        Uses claude-haiku-4-5-20251001 — fast and cost-effective for short prompts.
        max_tokens=200 is sufficient for a 2-3 sentence explanation.

        Returns:
            string — explanation text, or an error/fallback message
        """
        if self._claude_client is None:
            # No API key set — return a readable rule-based explanation instead
            return self._rule_based_explanation()

        try:
            response = self._claude_client.messages.create(
                model='claude-haiku-4-5-20251001',
                max_tokens=200,
                messages=[{'role': 'user', 'content': prompt}],
            )
            # response.content is a list of content blocks; [0].text is the text
            return response.content[0].text

        except Exception as e:
            # Return a safe fallback instead of crashing predict()
            return f'[Claude API error: {e}]'

    def _rule_based_explanation(self):
        """
        Generates a plain-text explanation from stored attribution data.

        Called automatically when ANTHROPIC_API_KEY is not set.
        Reads self._last_* attributes populated by _build_prompt().

        Returns a 2-3 sentence string that is honest and useful without an API call.
        """
        # These are set by _build_prompt() before _call_claude() is invoked
        label      = getattr(self, '_last_label',      'UNKNOWN')
        confidence = getattr(self, '_last_confidence', 0)
        top_words  = getattr(self, '_last_top_words',  [])
        num_lines  = getattr(self, '_last_num_lines',  [])

        direction = 'true' if label == 'TRUE' else 'false'

        parts = []

        # Sentence 1: verdict + confidence
        parts.append(
            f'This statement is likely {direction} (confidence: {confidence}%).'
        )

        # Sentence 2: most influential words (only if we have some)
        if top_words:
            word_list = ', '.join(top_words)
            parts.append(
                f'The words most associated with this decision were: {word_list}.'
            )

        # Sentence 3: strongest speaker/context signal (first num_lines entry)
        # num_lines entries look like: '   - the speaker has been truthful 30% ...'
        # We strip leading spaces and the dash before capitalising.
        if num_lines:
            # strip whitespace and leading '- ' from the evidence line
            evidence = num_lines[0].strip().lstrip('- ')
            # Capitalise first letter and add full stop if not already present
            evidence = evidence[0].upper() + evidence[1:]
            if not evidence.endswith('.'):
                evidence += '.'
            parts.append(evidence)

        return ' '.join(parts)

    # ── Main explain method ───────────────────────────────────────────────────

    def explain(self, prep, prob, label, n_steps=20, n_samples=50):
        """
        Full explanation pipeline for one prediction.

        Args:
            prep      : dict from _prepare_input() in predict.py
            prob      : P(TRUE) float in [0, 1]
            label     : 'TRUE' or 'FALSE'
            n_steps   : LayerIG integration steps (20 for CPU, 50+ for GPU)
            n_samples : GradientSHAP samples (50 for CPU, 100 for GPU)

        Returns:
            explanation : plain text string

        Approximate timing on CPU:
            n_steps=20, n_samples=50 → ~60-90 seconds total
        """
        # Step 1: LayerIG for text attribution
        tok_attrs, delta = self._explain_text(prep, n_steps)

        # Completeness check — is the IG approximation good enough?
        # With n_steps=20-50, delta < 0.20 is acceptable (not a bug — just Riemann approx.)
        # Only delta >= 0.20 indicates a real convergence problem → use LIME instead.
        use_ig = delta < 0.20

        if delta < 0.05:
            print(f'  IG convergence: EXCELLENT (delta={delta:.4f})')
        elif delta < 0.20:
            print(f'  IG convergence: ACCEPTABLE (delta={delta:.4f})')
        else:
            print(f'  IG convergence: POOR (delta={delta:.4f}) — falling back to LIME')

        # Step 2: LIME fallback if IG did not converge
        if not use_ig:
            print('  Running LIME fallback...')
            tok_attrs = self._explain_text_lime(prep)

        # Step 3: GradientSHAP for numerical features
        num_attrs = self._explain_numerical(prep, n_samples)

        # Step 4: Build Claude prompt and get explanation
        prompt      = self._build_prompt(prep, prob, label, tok_attrs, num_attrs, use_ig)
        explanation = self._call_claude(prompt)

        return explanation
