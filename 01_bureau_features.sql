-- =====================================================================
-- 01_bureau_features.sql
-- One row per applicant (SK_ID_CURR) summarising their credit history
-- at OTHER lenders, as reported by the credit bureau.
--
-- Leakage check: every bureau record describes a loan that existed
-- before the current application (DAYS_CREDIT is always <= 0), so these
-- features are known at the time of the lending decision.
-- =====================================================================
CREATE OR REPLACE TABLE bureau_features AS
SELECT
    SK_ID_CURR,

    -- how much credit history the applicant has
    COUNT(*)                                                        AS BUR_LOAN_COUNT,
    SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END)       AS BUR_ACTIVE_COUNT,
    SUM(CASE WHEN CREDIT_ACTIVE = 'Closed' THEN 1 ELSE 0 END)       AS BUR_CLOSED_COUNT,
    SUM(CASE WHEN CREDIT_ACTIVE IN ('Bad debt', 'Sold') THEN 1 ELSE 0 END) AS BUR_BAD_DEBT_COUNT,
    SUM(CASE WHEN CREDIT_TYPE = 'Credit card' THEN 1 ELSE 0 END)    AS BUR_CREDIT_CARD_COUNT,
    SUM(CASE WHEN CREDIT_TYPE = 'Microloan' THEN 1 ELSE 0 END)      AS BUR_MICROLOAN_COUNT,

    -- how old / how recent the history is (days, negative = in the past)
    MIN(DAYS_CREDIT)                                                AS BUR_OLDEST_CREDIT_DAYS,
    MAX(DAYS_CREDIT)                                                AS BUR_NEWEST_CREDIT_DAYS,
    SUM(CASE WHEN DAYS_CREDIT >= -365 THEN 1 ELSE 0 END)            AS BUR_LOANS_LAST_YEAR,

    -- current indebtedness
    SUM(COALESCE(AMT_CREDIT_SUM, 0))                                AS BUR_TOTAL_CREDIT,
    SUM(COALESCE(AMT_CREDIT_SUM_DEBT, 0))                           AS BUR_TOTAL_DEBT,
    SUM(COALESCE(AMT_CREDIT_SUM_DEBT, 0))
        / NULLIF(SUM(COALESCE(AMT_CREDIT_SUM, 0)), 0)               AS BUR_DEBT_TO_CREDIT,

    -- repayment behaviour
    MAX(COALESCE(CREDIT_DAY_OVERDUE, 0))                            AS BUR_MAX_DAYS_OVERDUE,
    MAX(COALESCE(AMT_CREDIT_MAX_OVERDUE, 0))                        AS BUR_MAX_AMT_OVERDUE,
    SUM(COALESCE(AMT_CREDIT_SUM_OVERDUE, 0))                        AS BUR_TOTAL_AMT_OVERDUE,
    MAX(CASE WHEN COALESCE(CREDIT_DAY_OVERDUE, 0) > 0 THEN 1 ELSE 0 END) AS BUR_HAS_OVERDUE,
    SUM(COALESCE(CNT_CREDIT_PROLONG, 0))                            AS BUR_PROLONG_COUNT
FROM bureau
GROUP BY SK_ID_CURR;
