-- =====================================================================
-- 02_previous_application_features.sql
-- One row per applicant summarising their PREVIOUS applications with
-- Home Credit itself: how often they were approved / refused, how much
-- they asked for vs. how much they actually got.
--
-- Leakage check: DAYS_DECISION is relative to the current application
-- and is always negative, so these are past events only.
-- =====================================================================
CREATE OR REPLACE TABLE prev_features AS
SELECT
    SK_ID_CURR,

    COUNT(*)                                                             AS PREV_APP_COUNT,
    SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1 ELSE 0 END)   AS PREV_APPROVED_COUNT,
    SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused'  THEN 1 ELSE 0 END)   AS PREV_REFUSED_COUNT,
    AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Refused'  THEN 1.0 ELSE 0.0 END) AS PREV_REFUSAL_RATE,

    AVG(AMT_APPLICATION)                                                 AS PREV_AVG_AMT_APPLIED,
    AVG(AMT_CREDIT)                                                      AS PREV_AVG_AMT_CREDIT,
    -- < 1 means the lender gave less than the customer asked for
    AVG(AMT_CREDIT / NULLIF(AMT_APPLICATION, 0))                         AS PREV_AVG_GRANTED_RATIO,
    AVG(CNT_PAYMENT)                                                     AS PREV_AVG_TERM_MONTHS,

    MAX(DAYS_DECISION)                                                   AS PREV_LAST_DECISION_DAYS,
    -- share of previous loans priced in the "high" yield group (riskier customers)
    AVG(CASE WHEN NAME_YIELD_GROUP = 'high' THEN 1.0 ELSE 0.0 END)       AS PREV_HIGH_YIELD_SHARE
FROM previous_application
GROUP BY SK_ID_CURR;
