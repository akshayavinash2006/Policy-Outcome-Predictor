# Decision memo: Free electricity for agricultural pumps
*Policy Outcome Predictor · 21 Nov 2026 · decision support, not a decision*

## 1. Bottom line
**Most likely outcome: Mixed** (57%), confidence **medium** (evidence-weighted reference class).

| Outcome | Probability | Corpus base rate |
|---|---|---|
| Improved | 21% | 29% |
| Mixed | 57% | 41% |
| No clear effect | 11% | 10% |
| Worsened | 11% | 20% |

Net-outcome index: **+0.10** (on a scale from -1, certain to worsen, to +1, certain to improve). Reference class: 8 precedents, effective sample size 4.5, best match similarity 0.63 (relevance gate 0.42).

## 2. Strategic lens: Commons / externality
*Assigned by: knn.* Similarity-weighted vote of nearest precedents.

Users of a shared resource (groundwater, air, roads, the grid) don't bear the full social cost of their use, so they over-use it. Policy must put a price on, ration or assign rights to the resource.

**Players:** Individual users <-> other users; regulator; enforcement agency

**What typically goes wrong under this lens**
- Over-extraction continues when use is unpriced or unmetered
- Substitution and displacement (second cars, shifting to other hours)
- Rebound effects: efficiency gains increase total use
- Weak enforcement turns rules into suggestions
- Free-riding on voluntary schemes

**Diagnostic questions for the policy team**
- Is individual use observable and measurable?
- Can users substitute around the rule?
- Who enforces, and what does non-compliance cost the user?

*Theory: Hardin (1968) [5]; Ostrom (1990)*

## 3. Reference class (precedents)
| # | Precedent | Year | Similarity | Weight | Outcome | Evidence | Shared | Differs |
|---|---|---|---|---|---|---|---|---|
| 1 | [Jyotigram Yojana (feeder separation)](https://www.worldbank.org/en/events/2018/03/01/impact-of-farm-electricity-supply-management-on-farm-households-evidence-from-a-natural-experiment-in-india) | n/a | 0.62 | 0.57 | Mixed | quasi_experimental/medium | sector: energy_water_environment; strategic lens: commons_externality; level of government: state | mechanism: subsidy vs infrastructure; state: Punjab vs Gujarat |
| 2 | [PM-KUSUM (solar pumps and decentralised solar)](https://prsindia.org/policy/report-summaries/implementation-of-pm-kusum-and-pm-surya-ghar) | n/a | 0.63 | 0.45 | Mixed | official_data/medium | sector: energy_water_environment; mechanism: subsidy | strategic lens: commons_externality vs principal_agent; level of government: state vs central |
| 3 | [Perform Achieve and Trade (PAT) scheme](https://www.beeindia.gov.in/sites/default/files/press_releases/Brief%20Note%20on%20PAT%20Scheme.pdf) | n/a | 0.57 | 0.13 | Improved | official_data/medium | sector: energy_water_environment; strategic lens: commons_externality | mechanism: subsidy vs quota; level of government: state vs central |
| 4 | [Perform Achieve and Trade (PAT) scheme](https://www.cseindia.org/content/downloadreports/11070) | n/a | 0.56 | 0.12 | Mixed | study/medium | sector: energy_water_environment; strategic lens: commons_externality | mechanism: subsidy vs quota; level of government: state vs central |
| 5 | [Jyotigram Yojana (feeder separation)](https://www.iwmi.cgiar.org/Publications/Success_Stories/PDF/2011/Issue_9-Innovative electricity scheme sparks.pdf) | n/a | 0.55 | 0.05 | Improved | study/low | sector: energy_water_environment; strategic lens: commons_externality; level of government: state | mechanism: subsidy vs infrastructure; state: Punjab vs Gujarat |
| 6 | [Jalyukt Shivar Abhiyan](https://www.indiawaterportal.org/drinking-water/jalyukt-shivar-abhiyan-low-impact-high-expenditure-scheme-cag) | 2014 | 0.59 | 0.20 | No clear effect | audit/medium | sector: energy_water_environment; level of government: state | mechanism: subsidy vs infrastructure; strategic lens: commons_externality vs principal_agent; state: Punjab vs Maharashtra |
| 7 | [Pradhan Mantri Ujjwala Yojana (PMUY)](https://www.downtoearth.org.in/news/governance/lpg-connections-not-only-success-metric-cag-on-ujjwala-68387) | 2016 | 0.52 | 0.08 | Mixed | audit/high | sector: energy_water_environment; mechanism: subsidy | strategic lens: commons_externality vs coordination; level of government: state vs central |
| 8 | [Punjab Preservation of Subsoil Water Act (paddy transplanting date ban)](https://econpapers.repec.org/paper/ashwpaper/61.htm) | 2009 | 0.51 | 0.05 | Mixed | quasi_experimental/medium | strategic lens: commons_externality; level of government: state | sector: energy_water_environment vs agriculture; mechanism: subsidy vs regulation |

- **Jyotigram Yojana (feeder separation)** `[jyotigram_yojana_feeder_separation_1]`: Difference-in-differences on IHDS 2004-05 and 2011-12: no increase in electric pump ownership; diesel pump ownership fell; tubewell ownership and tubewell irrigation rose; per-acre cost of purchased irrigation water rose; net farm income per acre fell on average (more for medium-large farmers); yields not lower *Side effects:* Higher water purchase costs for farmers; water markets shifted
- **PM-KUSUM (solar pumps and decentralised solar)** `[pm_kusum_solar_pumps_and_decentralised_s_1]`: Estimates Committee (Dec 2025): Component A achieved 641 MW vs 10,000 MW target (July 2025); Component C only 60,828 pumps sanctioned vs 1.5 lakh target *Side effects:* No CFA for Comp A and land conversion costs deter farmers; under Comp C farmers bear ~70% cost where states add nothing; high evacuation costs
- **Perform Achieve and Trade (PAT) scheme** `[perform_achieve_and_trade_pat_scheme_1]`: Cycle I (to Mar 2015): 478 units in 8 sectors saved 8.67 Mtoe vs 6.686 Mtoe target (+30%), ~31 Mt CO2 avoided, ~13 lakh ESCerts traded (~Rs 100 crore). Cycle II: 621 DCs saved ~13.28 Mtoe vs 8.869 Mtoe target, ~61.34 Mt CO2 avoided *Side effects:* Lenient targets (see CSE row); over-achievement may reflect soft baselines
- **Perform Achieve and Trade (PAT) scheme** `[perform_achieve_and_trade_pat_scheme_2]`: CSE (2021): TPPs saved 3.06 Mtoe vs 3.211 Mtoe target in Cycle I (~5% short) and 3.519 vs 3.13 Mtoe in Cycle II; CO2 cut only ~1.4-1.6% of power-sector emissions; 21 (C-I) and 46 (C-II) plants complied by buying ESCerts; ~80% of TPP targets laxer than CEA benchmarks; no TPP penalised *Side effects:* Cheap ESCerts (~Rs 700 avg vs ~Rs 4,020/toe abatement cost at one plant) made buying cheaper than saving; Cycle II ESCert surplus 5.84 lakh
- **Jyotigram Yojana (feeder separation)** `[jyotigram_yojana_feeder_separation_2]`: IWMI reports the scheme reduced farm power subsidy and capped groundwater overdraft (no quantities given); Gujarat agricultural growth ~10% p.a. 2000/01-2006/07 vs 2.9% nationally (correlational, not attributed causally) *Side effects:* Not quantified in source

## 4. What separated success from failure among the precedents
**Improved:**
- Perform Achieve and Trade (PAT) scheme: Plant-specific energy-intensity targets; over-achievers earn tradable Energy Saving Certificates (ESCerts), under-achievers buy them or pay penalty (quota)
- Jyotigram Yojana (feeder separation): USD 260 million invested to separate agricultural and non-agricultural feeders (infrastructure)

**Worsened or no clear effect:**
- Jalyukt Shivar Abhiyan: Decentralised water-harvesting works (deepening, check dams, desilting) in villages (infrastructure)

## 5. Risk register
| Risk observed in a precedent | Precedent outcome | Evidence | Source |
|---|---|---|---|
| Higher water purchase costs for farmers; water markets shifted | Mixed | quasi_experimental/medium | `jyotigram_yojana_feeder_separation_1` |
| No CFA for Comp A and land conversion costs deter farmers; under Comp C farmers bear ~70% cost where states add nothing; high evacuation costs | Mixed | official_data/medium | `pm_kusum_solar_pumps_and_decentralised_s_1` |
| CAG flagged lack of transparency (per The Wire headline) | No clear effect | audit/medium | `jalyukt_shivar_abhiyan_1` |
| Lenient targets (see CSE row); over-achievement may reflect soft baselines | Improved | official_data/medium | `perform_achieve_and_trade_pat_scheme_1` |
| Cheap ESCerts (~Rs 700 avg vs ~Rs 4,020/toe abatement cost at one plant) made buying cheaper than saving; Cycle II ESCert surplus 5.84 lakh | Mixed | study/medium | `perform_achieve_and_trade_pat_scheme_2` |
| Seven of 11 states below-average consumption (Chhattisgarh, MP, Jharkhand, Odisha, Assam, WB, Maharashtra) | Mixed | audit/high | `pradhan_mantri_ujjwala_yojana_pmuy_2` |

## 6. Design levers to consider

*Standard levers for the commons / externality lens:*
- Meter and price use (Pigouvian pricing, volumetric tariffs)
- Tradable permits or certificates (cap-and-trade, PAT ESCerts)
- Separate the resource physically (e.g. feeder separation) to make rationing enforceable
- Pay for conservation relative to a baseline
- Community-level monitoring and graduated sanctions (Ostrom)

## Limitations
- The corpus holds 57 records; probabilities are coarse and should be read as ranges.
- Outcome labels summarise heterogeneous metrics; 'mixed' is the most common label.
- Precedents are associations, not causal estimates for this policy's context.
- Lens tags are the project's classification, not claims made by the sources.