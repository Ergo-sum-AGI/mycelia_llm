# ============================================
# FILE: mycelia_jupyter_logger.py
# Save this in /home/ec2-user/SageMaker/
# ============================================

import sys

# Try importing IPython display, fallback to dummy function if not available
try:
    from IPython.display import clear_output
    _HAS_IPYTHON = True
except ImportError:
    _HAS_IPYTHON = False
    def clear_output(*args, **kwargs):
        pass


class MyceliaJupyterLogger:
    def __init__(self, refresh_rate_steps: int = 10):
        self.refresh_rate = refresh_rate_steps
        self.GREEN = "🟩"
        self.RED = "🟥"

    def update(self, step: int, stats_dict: dict, info_dict: dict):
        """Update the dashboard with consensus and compression telemetry."""
        if step % self.refresh_rate != 0:
            return

        total = stats_dict.get('total', 0)
        if total == 0:
            return

        kept = stats_dict.get('kept', 0)
        vetoed = stats_dict.get('vetoed', 0)
        kept_pct = (kept / total) * 100 if total > 0 else 0
        vetoed_pct = (vetoed / total) * 100 if total > 0 else 0

        # Status bar
        kept_bars = min(10, round(kept_pct / 10))
        vetoed_bars = 10 - kept_bars
        bar_visual = f"[{self.GREEN * kept_bars}{self.RED * vetoed_bars}]"

        # Build output
        output_str = (
            f"\n🍄 MYCELIA TELEMETRY DASHBOARD 🍄\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Step: {step:06d}\n"
            f"───────────────────────────────────────────────────\n"
            f"Consensus State : {bar_visual}\n"
            f"Kept Tokens     : {kept_pct:.2f}%\n"
            f"Vetoed Tokens   : {vetoed_pct:.2f}%\n"
            f"───────────────────────────────────────────────────\n"
            f"Coherence Score : {info_dict.get('coherence', 0):.4f}\n"
            f"Max Variance    : {info_dict.get('variance', 0):.5f}\n"
            f"Veto Threshold  : {info_dict.get('threshold', 0):.3f}\n"
        )
        
        # ─── ADD THREE-STATE TELEMETRY DISPLAY HERE ─────────────────────────────
        # Display the three-state distribution if available
        if 'telemetry_stats' in info_dict:
            stats = info_dict['telemetry_stats']
            output_str += (
                f"\n───────────────────────────────────────────────────\n"
                f"📊 TELEMETRY STATE DISTRIBUTION\n"
                f"🟩 Safe (Var ≤ 2.5):    {stats.get('safe_pct', 0):.1f}%\n"
                f"🟨 Dissenter (2.5-7.0): {stats.get('dissenter_pct', 0):.1f}%\n"
                f"🟥 Dubito (Var > 7.0):  {stats.get('dubito_pct', 0):.1f}%\n"
            )
        # ──────────────────────────────────────────────────────────────────────────

        # Add compression metrics if they exist in info_dict
        if 'compress_ratio' in info_dict:
            output_str += (
                f"\n───────────────────────────────────────────────────\n"
                f"⚡ COMPRESSION METRICS\n"
                f"Compression Ratio   : {info_dict.get('compress_ratio', 8)}:1\n"
                f"Active Batch Savings: {info_dict.get('vram_saved', 0.0):.2f} MB\n"
                f"Total Run Savings   : {info_dict.get('cumulative_gb', 0.0):.3f} GB ✨\n"
            )

        # ─── ADD SWEET SPOT SCORE DISPLAY ───────────────────────────────────────
        if 'sweet_spot_score' in info_dict:
            output_str += (
                f"\n───────────────────────────────────────────────────\n"
                f"🎯 ARCHITECTURE OPTIMIZATION TARGET\n"
                f"Sweet Spot Score: {info_dict.get('sweet_spot_score', 0.0):.4f}"
            )
            if info_dict.get('sweet_spot_score', 0.0) > 1.5:
                output_str += " ⭐ (OPTIMAL!)"
            elif info_dict.get('sweet_spot_score', 0.0) > 1.0:
                output_str += " ✅ (Good)"
            elif info_dict.get('sweet_spot_score', 0.0) > 0.5:
                output_str += " ⚠️ (Suboptimal)"
            else:
                output_str += " ❌ (Poor - tune compression)"

        if vetoed_pct > 60.0:
            output_str += f"\n⚠️  HIGH DISSENT: {vetoed_pct:.1f}% vetoed!"

        output_str += f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        if _HAS_IPYTHON:
            clear_output(wait=True)
        sys.stdout.write(output_str + "\n")
        sys.stdout.flush()