-- =====================================================================
-- 04_portfolio_analysis.sql
-- Business questions answered directly in SQL on the model table.
-- Each query starts with "-- name: <query_name>" so src/eda.py can run
-- them one by one and save each result as a CSV + chart.
-- =====================================================================

-- name: portfolio_summary
SELECT
    COUNT(*)                                  AS applications,
    SUM(TARGET)                               AS bad_loans,
    ROUND(AVG(TARGET) * 100, 2)               AS bad_rate_pct,
    ROUND(SUM(AMT_CREDIT) / 1e6, 1)           AS total_credit_mn,
    ROUND(AVG(AMT_CREDIT), 0)                 AS avg_loan_size,
    ROUND(MEDIAN(AMT_INCOME_TOTAL), 0)        AS median_income
FROM model_table;

-- name: bad_rate_by_income_decile
WITH ranked AS (
    SELECT *, NTILE(10) OVER (ORDER BY AMT_INCOME_TOTAL) AS income_decile
    FROM model_table
)
SELECT
    income_decile,
    ROUND(MIN(AMT_INCOME_TOTAL), 0)  AS min_income,
    ROUND(MAX(AMT_INCOME_TOTAL), 0)  AS max_income,
    COUNT(*)                         AS applications,
    ROUND(AVG(TARGET) * 100, 2)      AS bad_rate_pct
FROM ranked
GROUP BY income_decile
ORDER BY income_decile;

-- name: bad_rate_by_credit_to_income
SELECT
    CASE
        WHEN CREDIT_INCOME_RATIO < 1 THEN '1: < 1x income'
        WHEN CREDIT_INCOME_RATIO < 2 THEN '2: 1-2x'
        WHEN CREDIT_INCOME_RATIO < 3 THEN '3: 2-3x'
        WHEN CREDIT_INCOME_RATIO < 5 THEN '4: 3-5x'
        ELSE '5: 5x+'
    END                              AS credit_to_income_band,
    COUNT(*)                         AS applications,
    ROUND(AVG(TARGET) * 100, 2)      AS bad_rate_pct,
    ROUND(AVG(AMT_CREDIT), 0)        AS avg_loan_size
FROM model_table
GROUP BY 1
ORDER BY 1;

-- name: bad_rate_by_emi_burden
SELECT
    CASE
        WHEN ANNUITY_INCOME_RATIO < 0.10 THEN '1: < 10%'
        WHEN ANNUITY_INCOME_RATIO < 0.20 THEN '2: 10-20%'
        WHEN ANNUITY_INCOME_RATIO < 0.30 THEN '3: 20-30%'
        ELSE '4: 30%+'
    END                              AS emi_to_income_band,
    COUNT(*)                         AS applications,
    ROUND(AVG(TARGET) * 100, 2)      AS bad_rate_pct
FROM model_table
GROUP BY 1
ORDER BY 1;

-- name: bad_rate_by_external_score_decile
WITH ranked AS (
    SELECT *, NTILE(10) OVER (ORDER BY EXT_SOURCE_MEAN) AS score_decile
    FROM model_table
    WHERE EXT_SOURCE_MEAN IS NOT NULL
)
SELECT
    score_decile,
    ROUND(AVG(EXT_SOURCE_MEAN), 3)   AS avg_score,
    COUNT(*)                         AS applications,
    ROUND(AVG(TARGET) * 100, 2)      AS bad_rate_pct
FROM ranked
GROUP BY score_decile
ORDER BY score_decile;

-- name: bad_rate_by_bureau_history
SELECT
    CASE
        WHEN HAS_BUREAU_HISTORY = 0 THEN '1: no bureau record'
        WHEN BUR_HAS_OVERDUE = 1    THEN '3: currently overdue'
        ELSE '2: bureau record, no overdue'
    END                              AS bureau_status,
    COUNT(*)                         AS applications,
    ROUND(AVG(TARGET) * 100, 2)      AS bad_rate_pct
FROM model_table
GROUP BY 1
ORDER BY 1;

-- name: bad_rate_by_previous_refusals
SELECT
    CASE
        WHEN PREV_APP_COUNT = 0       THEN '1: new customer'
        WHEN PREV_REFUSED_COUNT = 0   THEN '2: never refused'
        WHEN PREV_REFUSED_COUNT <= 2  THEN '3: refused 1-2 times'
        ELSE '4: refused 3+ times'
    END                              AS refusal_history,
    COUNT(*)                         AS applications,
    ROUND(AVG(TARGET) * 100, 2)      AS bad_rate_pct
FROM model_table
GROUP BY 1
ORDER BY 1;

-- name: bad_rate_by_income_type
SELECT
    NAME_INCOME_TYPE,
    COUNT(*)                         AS applications,
    ROUND(AVG(TARGET) * 100, 2)      AS bad_rate_pct,
    -- share of the whole book, using a window function over the grouped rows
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS share_of_book_pct
FROM model_table
GROUP BY NAME_INCOME_TYPE
HAVING COUNT(*) >= 50
ORDER BY bad_rate_pct DESC;

-- name: bad_rate_by_age_band
SELECT
    CASE
        WHEN AGE_YEARS < 25 THEN '1: < 25'
        WHEN AGE_YEARS < 35 THEN '2: 25-34'
        WHEN AGE_YEARS < 45 THEN '3: 35-44'
        WHEN AGE_YEARS < 55 THEN '4: 45-54'
        ELSE '5: 55+'
    END                              AS age_band,
    COUNT(*)                         AS applications,
    ROUND(AVG(TARGET) * 100, 2)      AS bad_rate_pct
FROM model_table
GROUP BY 1
ORDER BY 1;
