from master_thesis.llm.schemas import (
    Trial
)


def build_trial(trial: Trial) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": trial.system_prompt},
        {
            "role": "user",
            "content": (
                f"prior: {trial.prior_prompt}\n"
                f"xs: {trial.xs}\n"
                f"observations: {trial.observations}\n"
                f"H0: {trial.H0}\n"
                f"H1: {trial.H1}\n"
                f"H0: BIC = {trial.bic_h0}, SSE = {trial.sse_h0}\n"
                f"H1: BIC = {trial.bic_h1}, SSE = {trial.sse_h1}\n"
                f"Delta BIC = {trial.delta_bic}"
            )
        }
    ]