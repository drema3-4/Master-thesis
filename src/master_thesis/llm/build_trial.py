from master_thesis.llm.schemas import (
    Trial
)


def build_trial(trial: Trial) -> list[dict[str, str]]:
    hypotheses = {
        "H0": (
            f"H0: {trial.H0}\n"
            f"H0: SSE = {trial.sse_h0}, BIC = {trial.bic_h0}, "
            f"log_likelihood = {trial.log_likelihood_h0}"
        ),
        "H1": (
            f"H1: {trial.H1}\n"
            f"H1: SSE = {trial.sse_h1}, BIC = {trial.bic_h1}, "
            f"log_likelihood = {trial.log_likelihood_h1}"
        )
    }
    hypothesis_order = trial.hypothesis_order.split("_")
    hypotheses_block = "\n".join(
        hypotheses[hypothesis]
        for hypothesis in hypothesis_order
    )

    return [
        {"role": "system", "content": trial.system_prompt},
        {
            "role": "user",
            "content": (
                f"prior: {trial.prior_prompt}\n"
                f"xs: {trial.xs}\n"
                f"observations - f(xs): {trial.observations}\n"
                f"{hypotheses_block}\n"
                f"Delta BIC = {trial.delta_bic}\n"
                f"log_likelihood_ratio = {trial.log_likelihood_ratio}"
            )
        }
    ]
