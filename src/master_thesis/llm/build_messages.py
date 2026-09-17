from master_thesis.schemas.ex1.trial_schema import (
    TrialSchema
)


SYSTEM = (
    "Ты анализируешь результаты эксперимента.\n"
    "Есть:\n"
    "1. полученные наблюдения: xs и f(xs) - observations, соответствующие какому-то неизвестному закону;\n"
    "2. гипотеза 1 - H0 и закон, который ей соответствует;\n"
    "3. гипотеза 2 - H1 и закон, который ей соответствует;\n"
    "4. метрики BIC и SSE гипотезы H0;\n"
    "5. метрики BIC и SSE гипотезы H1;\n"
    "6. Delta BIC = BIC_h1 - BIC_h0.\n"
    "Ты должен понять по представленным данным, гипотезам и метрикам, какая гипотеза "
    "наилучшим образом описывает данные. Другими словами какой закон верен.\n"
    "В качестве ответа всегда выводи коротко H0 или H1, в зависимости от того, "
    "какую гипотезу ты выбрал."
)


def build_messages(trial: TrialSchema) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
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