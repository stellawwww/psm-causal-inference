# Background: where this dataset comes from, and why it became the standard test

Every number quoted here is reproduced by the code in this repository. The benchmark comes
from `outputs/data/01_benchmark.json`, the naive comparisons from `outputs/tables/results.csv`.

---

## The policy problem, 1975

In the mid-1970s the open question in US labor policy was whether government could do anything
for the hardest-to-employ. Not people between jobs, but people facing severe structural
barriers: long-term welfare recipients, ex-offenders, recovering addicts, and teenagers who had
left school and never held steady work.

The National Supported Work Demonstration was built on a specific theory of why these groups
stayed unemployed. The claim was that the binding constraint was not skills. It was the absence
of any bridge into the labor market: no work history, no references, no experience with showing
up daily, and no employer willing to tolerate a slow start.

So the program offered 9 to 18 months of paid work in a deliberately supportive setting, with
peer groups, close supervision, and demands that ramped up gradually. It ran in ten cities,
funded by the Ford Foundation together with several federal agencies and administered by MDRC.

The question it had to answer was narrow and expensive: **does this raise participants' earnings
later, by enough to justify the cost?**

## Why they randomized

More people applied than there were places. Rather than admit on a first-come basis or by
caseworker judgment, the researchers assigned admission by lottery among eligible applicants.
That single decision is the reason anyone still studies this program fifty years later.

Randomization makes the two arms comparable on everything, including characteristics nobody
measured. You can see it directly in the data, and notebook 01 prints the table:

| | Treated | Randomized control | Difference |
|---|---|---|---|
| Age | 25.82 | 25.05 | 0.77 years |
| Education | 10.35 | 10.09 | 0.26 years |
| Black | 84% | 83% | 1 point |

Nobody adjusted anything to produce that. The lottery did it.

## What the experiment found

Because assignment was random, the effect estimate requires no modelling at all. Subtract the
means.

| | 1978 earnings |
|---|---|
| Treated (n=185) | $6,349 |
| Randomized control (n=260) | $4,555 |
| **Difference** | **$1,794**, 95% CI [474, 3,115] |

This is the answer key. Everything else in this repository is an attempt to recover it without
being allowed to look at it.

## LaLonde's actual question, 1986

Here is the part that motivates the project, and it is methodological rather than about labor
policy.

By 1986, almost no social program was evaluated by randomized trial. Randomization was
expensive, slow, and frequently impossible politically. Evaluators instead used observational
data with econometric adjustment: regression controls, fixed effects, difference-in-differences,
Heckman selection correction. Decisions worth billions of dollars rested on those estimates.

Robert LaLonde asked whether that trust was warranted. His test design was simple and brutal:

1. Take a program whose true effect is known from an experiment.
2. Throw the randomized control group away.
3. Build a comparison group the way an observational researcher would, by drawing ordinary
   people from a national survey.
4. Apply the standard econometric methods of the day.
5. See whether they recover the experimental answer.

They did not. The estimates scattered widely, frequently carried the wrong sign, and the
specification tests available at the time could not distinguish the good estimates from the bad
ones. The paper landed hard, and it is one reason randomized evaluation became the expected
standard in social policy over the following two decades.

This repository reproduces the core of his finding before any adjustment is applied:

| Comparison group | Naive difference in 1978 earnings |
|---|---|
| CPS survey respondents | **-$8,498** |
| PSID survey respondents | **-$15,205** |

Both say the program destroyed participants' earnings. Both are off by more than ten thousand
dollars, in the wrong direction.

## Why it became the standard test

Five properties, and it is rare for a dataset to have all of them at once.

- **There is an answer key.** Nearly every real dataset with real confounding has no ground
  truth, so a method can never be graded.
- **The confounding is genuine.** Survey respondents really are older, richer, better educated
  and mostly not Black compared to program participants. Nobody designed that bias, which means
  nobody can accidentally design a method that undoes it.
- **It is hard but not hopeless.** A test that every method passes, or that every method fails,
  teaches nothing.
- **It is small and fully legible.** Ten columns, each explainable in a sentence. No domain
  expertise is needed to reason about whether a variable is a confounder.
- **There is a literature to check against.** Four decades of published estimates on the same
  rows.

## What happened after

**Dehejia and Wahba (1999)** revisited the problem using propensity score matching, which
Rosenbaum and Rubin had introduced in 1983. They made two changes: they restricted the sample to
participants with two full years of pre-treatment earnings, which is the 185-row subsample in
`data/raw/lalonde/`, and they replaced regression adjustment with matching. They recovered
estimates close to the experimental benchmark. That paper is a large part of why propensity
score matching became a standard tool.

**Smith and Todd (2005)** pushed back, arguing the recovery was fragile. Change the subsample or
the specification in ways that are equally defensible, and performance degrades badly. Dehejia
replied. The exchange was never fully settled.

**That unsettled ending is the point of this project.** The canonical lesson is not "propensity
score matching works." It is:

> Propensity score matching works when the covariates capture the selection mechanism, and you
> cannot tell from inside the analysis whether they do.

This repository demonstrates it on the same data. Drop the 1974 and 1975 earnings variables and
the estimate moves from **+$1,752 to roughly -$3,000** — a sign flip — while the balance
diagnostics on the remaining covariates still look perfectly acceptable. The confounder you did
not measure is the one that decides your answer.

## References

- LaLonde, R. (1986). Evaluating the econometric evaluations of training programs with
  experimental data. *American Economic Review*, 76(4), 604-620.
- Dehejia, R. and Wahba, S. (1999). Causal effects in nonexperimental studies: reevaluating the
  evaluation of training programs. *Journal of the American Statistical Association*, 94(448),
  1053-1062.
- Smith, J. and Todd, P. (2005). Does matching overcome LaLonde's critique of nonexperimental
  estimators? *Journal of Econometrics*, 125(1-2), 305-353.
- Rosenbaum, P. and Rubin, D. (1983). The central role of the propensity score in observational
  studies for causal effects. *Biometrika*, 70(1), 41-55.

Data source: [Rajeev Dehejia's data page](https://users.nber.org/~rdehejia/data/).
