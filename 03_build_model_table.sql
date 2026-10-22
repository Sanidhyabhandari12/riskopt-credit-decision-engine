-- =====================================================================
-- 03_build_model_table.sql
-- Joins the application table with the two history summaries and
-- creates the business ratios used by the model.
-- Result: exactly ONE row per applicant (validated in Python afterwards).
-- =====================================================================
CREATE OR REPLACE TABLE model_table AS
WITH app AS (
    SELECT
        SK_ID_CURR,
        TARGET,
        NAME_CONTRACT_TYPE,
        NAME_INCOME_TYPE,
        NAME_EDUCATION_TYPE,
        NAME_FAMILY_STATUS,
        NAME_HOUSING_TYPE,
        OCCUPATION_TYPE,
        ORGANIZATION_TYPE,
        CASE WHEN FLAG_OWN_CAR = 'Y' THEN 1 ELSE 0 END    AS FLAG_OWN_CAR,
        CASE WHEN FLAG_OWN_REALTY = 'Y' THEN 1 ELSE 0 END AS FLAG_OWN_REALTY,
        CNT_CHILDREN,
        CNT_FAM_MEMBERS,
        AMT_INCOME_TOTAL,
        AMT_CREDIT,
        AMT_ANNUITY,
        AMT_GOODS_PRICE,
        REGION_POPULATION_RELATIVE,
        REGION_RATING_CLIENT_W_CITY,
        REG_CITY_NOT_WORK_CITY,
        DAYS_BIRTH,
        -- 365243 is a placeholder in the raw data (mostly pensioners / unemployed)
        CASE WHEN DAYS_EMPLOYED = 365243 THEN NULL ELSE DAYS_EMPLOYED END AS DAYS_EMPLOYED,
        CASE WHEN DAYS_EMPLOYED = 365243 THEN 1 ELSE 0 END               AS DAYS_EMPLOYED_ANOMALY,
        DAYS_REGISTRATION,
        DAYS_ID_PUBLISH,
        DAYS_LAST_PHONE_CHANGE,
        OWN_CAR_AGE,
        EXT_SOURCE_1,
        EXT_SOURCE_2,
        EXT_SOURCE_3,
        DEF_30_CNT_SOCIAL_CIRCLE,
        FLAG_DOCUMENT_3,
        AMT_REQ_CREDIT_BUREAU_YEAR
    FROM application
)
SELECT
    a.*,

    -- affordability / leverage ratios a credit analyst would look at
    a.AMT_CREDIT  / NULLIF(a.AMT_INCOME_TOTAL, 0)        AS CREDIT_INCOME_RATIO,
    a.AMT_ANNUITY / NULLIF(a.AMT_INCOME_TOTAL, 0)        AS ANNUITY_INCOME_RATIO,  -- EMI burden
    a.AMT_ANNUITY / NULLIF(a.AMT_CREDIT, 0)              AS ANNUITY_CREDIT_RATIO,  -- proxy for loan term
    a.AMT_CREDIT  / NULLIF(a.AMT_GOODS_PRICE, 0)         AS CREDIT_GOODS_RATIO,    -- LTV-like
    a.AMT_INCOME_TOTAL / NULLIF(a.CNT_FAM_MEMBERS, 0)    AS INCOME_PER_PERSON,
    -a.DAYS_BIRTH / 365.25                               AS AGE_YEARS,
    -a.DAYS_EMPLOYED / 365.25                            AS EMPLOYED_YEARS,

    -- external bureau scores are the strongest predictors in this dataset
    (COALESCE(a.EXT_SOURCE_1, 0) + COALESCE(a.EXT_SOURCE_2, 0) + COALESCE(a.EXT_SOURCE_3, 0))
        / NULLIF((CASE WHEN a.EXT_SOURCE_1 IS NULL THEN 0 ELSE 1 END)
               + (CASE WHEN a.EXT_SOURCE_2 IS NULL THEN 0 ELSE 1 END)
               + (CASE WHEN a.EXT_SOURCE_3 IS NULL THEN 0 ELSE 1 END), 0) AS EXT_SOURCE_MEAN,
    LEAST(a.EXT_SOURCE_1, a.EXT_SOURCE_2, a.EXT_SOURCE_3)                 AS EXT_SOURCE_MIN,

    -- bureau history (counts -> 0 when the applicant has no bureau record)
    CASE WHEN b.SK_ID_CURR IS NULL THEN 0 ELSE 1 END     AS HAS_BUREAU_HISTORY,
    COALESCE(b.BUR_LOAN_COUNT, 0)                        AS BUR_LOAN_COUNT,
    COALESCE(b.BUR_ACTIVE_COUNT, 0)                      AS BUR_ACTIVE_COUNT,
    COALESCE(b.BUR_CLOSED_COUNT, 0)                      AS BUR_CLOSED_COUNT,
    COALESCE(b.BUR_BAD_DEBT_COUNT, 0)                    AS BUR_BAD_DEBT_COUNT,
    COALESCE(b.BUR_CREDIT_CARD_COUNT, 0)                 AS BUR_CREDIT_CARD_COUNT,
    COALESCE(b.BUR_MICROLOAN_COUNT, 0)                   AS BUR_MICROLOAN_COUNT,
    COALESCE(b.BUR_LOANS_LAST_YEAR, 0)                   AS BUR_LOANS_LAST_YEAR,
    COALESCE(b.BUR_HAS_OVERDUE, 0)                       AS BUR_HAS_OVERDUE,
    COALESCE(b.BUR_PROLONG_COUNT, 0)                     AS BUR_PROLONG_COUNT,
    b.BUR_OLDEST_CREDIT_DAYS,
    b.BUR_NEWEST_CREDIT_DAYS,
    b.BUR_TOTAL_CREDIT,
    b.BUR_TOTAL_DEBT,
    b.BUR_DEBT_TO_CREDIT,
    b.BUR_MAX_DAYS_OVERDUE,
    b.BUR_MAX_AMT_OVERDUE,
    b.BUR_TOTAL_AMT_OVERDUE,
    b.BUR_TOTAL_DEBT / NULLIF(a.AMT_INCOME_TOTAL, 0)     AS BUR_DEBT_INCOME_RATIO,

    -- previous Home Credit applications
    CASE WHEN p.SK_ID_CURR IS NULL THEN 0 ELSE 1 END     AS HAS_PREV_APPLICATION,
    COALESCE(p.PREV_APP_COUNT, 0)                        AS PREV_APP_COUNT,
    COALESCE(p.PREV_APPROVED_COUNT, 0)                   AS PREV_APPROVED_COUNT,
    COALESCE(p.PREV_REFUSED_COUNT, 0)                    AS PREV_REFUSED_COUNT,
    p.PREV_REFUSAL_RATE,
    p.PREV_AVG_AMT_APPLIED,
    p.PREV_AVG_AMT_CREDIT,
    p.PREV_AVG_GRANTED_RATIO,
    p.PREV_AVG_TERM_MONTHS,
    p.PREV_LAST_DECISION_DAYS,
    p.PREV_HIGH_YIELD_SHARE
FROM app a
LEFT JOIN bureau_features b ON a.SK_ID_CURR = b.SK_ID_CURR
LEFT JOIN prev_features   p ON a.SK_ID_CURR = p.SK_ID_CURR;
