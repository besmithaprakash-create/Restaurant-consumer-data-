"""
Reproducible data-integrity audit for the
Restaurant & Consumer Data / Recommender Systems dataset.

Libraries used:
- Pandas
- NumPy

The audit checks:
- Restaurant information
- Restaurant cuisine
- Restaurant payment
- Opening hours
- Parking
- User cuisine preferences
- User payment preferences
- Missing values
- Duplicate records
- Unique restaurants and users
- Cross-dataset consistency
"""

from pathlib import Path
import re

import pandas as pd
import numpy as np


# ------------------------------------------------------------
# DATA LOCATION
# ------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent


# ------------------------------------------------------------
# LOAD CSV FILE
# ------------------------------------------------------------

def load_csv(required_columns):
    """
    Find a CSV file containing the required columns.
    """

    for file in sorted(DATA_DIR.glob("*.csv")):

        try:
            try:
                df=pd.read_csv(file,encoding="utf-8")
            except UnicodeDecodeError:
                df=pd.read_csv(file,encoding="latin1")
            df.columns = (
                df.columns
                .astype(str)
                .str.strip()
            )

            if set(required_columns).issubset(df.columns):
                return df

        except Exception:
            continue

    raise FileNotFoundError(
        f"CSV file not found for columns: {required_columns}"
    )


# ------------------------------------------------------------
# BASIC CHECKS
# ------------------------------------------------------------

def shape(df):
    """Return dataset shape."""

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1])
    }


def uniqueness(df, column):
    """Return number of unique values."""

    return int(
        df[column]
        .dropna()
        .nunique()
    )


def duplicates(df):
    """Return number of duplicate rows."""

    return int(
        df.duplicated().sum()
    )


def missingness(df):
    """Return missing values in every column."""

    result = {}

    for column in df.columns:

        result[column] = int(
            df[column].isna().sum()
        )

    return result


def question_mark_values(df):
    """
    Count '?' values used as missing values
    in the Restaurant & Consumer dataset.
    """

    result = {}

    for column in df.columns:

        result[column] = int(
            df[column]
            .astype(str)
            .str.strip()
            .eq("?")
            .sum()
        )

    return result


# ------------------------------------------------------------
# RESTAURANT INFORMATION AUDIT
# ------------------------------------------------------------

def restaurant_audit(df):

    result = {}

    result["shape"] = shape(df)

    result["unique_restaurants"] = uniqueness(
        df,
        "placeID"
    )

    result["duplicate_rows"] = duplicates(df)

    result["missingness"] = missingness(df)

    result["question_mark_values"] = (
        question_mark_values(df)
    )

    # Restaurant names
    if "name" in df.columns:

        names = (
            df["name"]
            .astype(str)
            .str.strip()
        )

        result["missing_names"] = int(
            names.isin(["", "?"]).sum()
        )

    # Latitude validation using NumPy
    if "latitude" in df.columns:

        latitude = pd.to_numeric(
            df["latitude"],
            errors="coerce"
        )

        result["invalid_latitude"] = int(
            np.sum(
                (latitude < -90)
                | (latitude > 90)
            )
        )

        result["missing_latitude"] = int(
            np.sum(latitude.isna())
        )

    # Longitude validation using NumPy
    if "longitude" in df.columns:

        longitude = pd.to_numeric(
            df["longitude"],
            errors="coerce"
        )

        result["invalid_longitude"] = int(
            np.sum(
                (longitude < -180)
                | (longitude > 180)
            )
        )

        result["missing_longitude"] = int(
            np.sum(longitude.isna())
        )

    # Price values
    if "price" in df.columns:

        result["price_values"] = sorted(
            df["price"]
            .astype(str)
            .str.strip()
            .replace("?", pd.NA)
            .dropna()
            .unique()
            .tolist()
        )

    # Alcohol values
    if "alcohol" in df.columns:

        result["alcohol_values"] = sorted(
            df["alcohol"]
            .astype(str)
            .str.strip()
            .replace("?", pd.NA)
            .dropna()
            .unique()
            .tolist()
        )

    return result


# ------------------------------------------------------------
# CUISINE AUDIT
# ------------------------------------------------------------

def cuisine_audit(
    df,
    id_column,
    cuisine_column
):
    """
    Works for:
    placeID,Rcuisine
    userID,Rcuisine
    """

    result = {}

    result["shape"] = shape(df)

    result["unique_ids"] = uniqueness(
        df,
        id_column
    )

    result["unique_cuisines"] = uniqueness(
        df,
        cuisine_column
    )

    result["duplicate_rows"] = duplicates(df)

    result["missingness"] = missingness(df)

    # Cuisine count per user/restaurant
    cuisines_per_id = (
        df.groupby(id_column)[cuisine_column]
        .nunique()
    )

    result["ids_with_multiple_cuisines"] = int(
        np.sum(
            cuisines_per_id.to_numpy() > 1
        )
    )

    if len(cuisines_per_id) > 0:

        result["maximum_cuisines"] = int(
            np.max(
                cuisines_per_id.to_numpy()
            )
        )

    else:

        result["maximum_cuisines"] = 0

    # Most common cuisines
    result["top_cuisines"] = (
        df[cuisine_column]
        .astype(str)
        .str.strip()
        .value_counts()
        .head(10)
        .to_dict()
    )

    return result


# ------------------------------------------------------------
# PAYMENT AUDIT
# ------------------------------------------------------------

def payment_audit(
    df,
    id_column,
    payment_column
):
    """
    Works for:
    placeID,Rpayment
    userID,Upayment
    """

    result = {}

    result["shape"] = shape(df)

    result["unique_ids"] = uniqueness(
        df,
        id_column
    )

    result["unique_payment_methods"] = uniqueness(
        df,
        payment_column
    )

    result["duplicate_rows"] = duplicates(df)

    result["missingness"] = missingness(df)

    payments_per_id = (
        df.groupby(id_column)[payment_column]
        .nunique()
    )

    result["ids_with_multiple_payments"] = int(
        np.sum(
            payments_per_id.to_numpy() > 1
        )
    )

    if len(payments_per_id) > 0:

        result["maximum_payment_methods"] = int(
            np.max(
                payments_per_id.to_numpy()
            )
        )

    else:

        result["maximum_payment_methods"] = 0

    result["payment_methods"] = (
        df[payment_column]
        .astype(str)
        .str.strip()
        .value_counts()
        .to_dict()
    )

    return result


# ------------------------------------------------------------
# PARKING AUDIT
# ------------------------------------------------------------

def parking_audit(df):

    result = {}

    result["shape"] = shape(df)

    result["unique_restaurants"] = uniqueness(
        df,
        "placeID"
    )

    result["unique_parking_types"] = uniqueness(
        df,
        "parking_lot"
    )

    result["duplicate_rows"] = duplicates(df)

    result["missingness"] = missingness(df)

    parking_per_restaurant = (
        df.groupby("placeID")["parking_lot"]
        .nunique()
    )

    result["restaurants_with_multiple_parking"] = int(
        np.sum(
            parking_per_restaurant.to_numpy() > 1
        )
    )

    result["parking_types"] = (
        df["parking_lot"]
        .astype(str)
        .str.strip()
        .value_counts()
        .to_dict()
    )

    expected_parking = {
        "none",
        "yes",
        "public",
        "fee",
        "valet parking",
        "street",
        "validated parking"
    }

    actual_parking = set(
        df["parking_lot"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace("?", pd.NA)
        .dropna()
    )

    result["unexpected_parking_values"] = sorted(
        actual_parking - expected_parking
    )

    return result


# ------------------------------------------------------------
# OPENING HOURS AUDIT
# ------------------------------------------------------------

def hours_audit(df):

    result = {}

    result["shape"] = shape(df)

    result["unique_restaurants"] = uniqueness(
        df,
        "placeID"
    )

    result["duplicate_rows"] = duplicates(df)

    result["missingness"] = missingness(df)

    invalid_time_rows = 0
    overnight_rows = 0
    full_day_rows = 0
    invalid_day_rows = 0

    valid_days = {
        "Mon",
        "Tue",
        "Wed",
        "Thu",
        "Fri",
        "Sat",
        "Sun"
    }

    time_pattern = re.compile(
        r"^\d{2}:\d{2}-\d{2}:\d{2}$"
    )

    for _, row in df.iterrows():

        hours = str(
            row["hours"]
        ).strip().rstrip(";")

        days = str(
            row["days"]
        ).strip().rstrip(";")

        # Check time format
        if not time_pattern.fullmatch(hours):

            invalid_time_rows += 1

        else:

            start, end = hours.split("-")

            try:

                start_hour, start_minute = map(
                    int,
                    start.split(":")
                )

                end_hour, end_minute = map(
                    int,
                    end.split(":")
                )

                valid_time_range = (
                    0 <= start_hour <= 23
                    and 0 <= end_hour <= 23
                    and 0 <= start_minute <= 59
                    and 0 <= end_minute <= 59
                )

                if not valid_time_range:

                    invalid_time_rows += 1

                else:

                    start_total = (
                        start_hour * 60
                        + start_minute
                    )

                    end_total = (
                        end_hour * 60
                        + end_minute
                    )

                    if (
                        start_total == 0
                        and end_total == 0
                    ):

                        full_day_rows += 1

                    elif end_total < start_total:

                        # Example:
                        # 21:00-01:00
                        overnight_rows += 1

            except ValueError:

                invalid_time_rows += 1

        # Check days
        day_list = [
            day.strip()
            for day in days.split(";")
            if day.strip()
        ]

        for day in day_list:

            if day not in valid_days:

                invalid_day_rows += 1
                break

    result["invalid_time_rows"] = (
        invalid_time_rows
    )

    result["overnight_rows"] = (
        overnight_rows
    )

    result["00_00_rows"] = (
        full_day_rows
    )

    result["invalid_day_rows"] = (
        invalid_day_rows
    )

    return result


# ------------------------------------------------------------
# USER ID AUDIT
# ------------------------------------------------------------

def user_id_audit(df):

    user_ids = (
        df["userID"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    invalid_ids = []

    for user_id in user_ids.unique():

        if not re.fullmatch(
            r"U\d+",
            user_id
        ):

            invalid_ids.append(user_id)

    return {
        "unique_users": int(
            user_ids.nunique()
        ),

        "invalid_user_ids": len(
            invalid_ids
        ),

        "invalid_examples": invalid_ids[:10]
    }


# ------------------------------------------------------------
# CROSS DATASET AUDIT
# ------------------------------------------------------------

def cross_dataset_audit(
    restaurant_df,
    restaurant_cuisine_df,
    restaurant_payment_df,
    hours_df,
    parking_df,
    user_cuisine_df,
    user_payment_df
):

    result = {}

    restaurant_ids = set(
        restaurant_df["placeID"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    cuisine_ids = set(
        restaurant_cuisine_df["placeID"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    payment_ids = set(
        restaurant_payment_df["placeID"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    hours_ids = set(
        hours_df["placeID"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    parking_ids = set(
        parking_df["placeID"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    # Restaurant counts
    result["restaurants"] = len(
        restaurant_ids
    )

    result["restaurants_with_cuisine"] = len(
        cuisine_ids
    )

    result["restaurants_with_payment"] = len(
        payment_ids
    )

    result["restaurants_with_hours"] = len(
        hours_ids
    )

    result["restaurants_with_parking"] = len(
        parking_ids
    )

    # Common restaurant IDs
    common_ids = (
        restaurant_ids
        & cuisine_ids
        & payment_ids
        & hours_ids
        & parking_ids
    )

    result["restaurants_common_to_all"] = len(
        common_ids
    )

    # Restaurant IDs missing from main file
    result["cuisine_ids_not_in_main"] = len(
        cuisine_ids - restaurant_ids
    )

    result["payment_ids_not_in_main"] = len(
        payment_ids - restaurant_ids
    )

    result["hours_ids_not_in_main"] = len(
        hours_ids - restaurant_ids
    )

    result["parking_ids_not_in_main"] = len(
        parking_ids - restaurant_ids
    )

    # User IDs
    user_cuisine_ids = set(
        user_cuisine_df["userID"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    user_payment_ids = set(
        user_payment_df["userID"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    result["users_with_cuisine"] = len(
        user_cuisine_ids
    )

    result["users_with_payment"] = len(
        user_payment_ids
    )

    result["users_common_to_both"] = len(
        user_cuisine_ids
        & user_payment_ids
    )

    result["users_without_payment"] = len(
        user_cuisine_ids
        - user_payment_ids
    )

    result["users_without_cuisine"] = len(
        user_payment_ids
        - user_cuisine_ids
    )

    return result


# ------------------------------------------------------------
# RECOMMENDER SYSTEM AUDIT
# ------------------------------------------------------------

def recommender_audit(
    restaurant_cuisine_df,
    user_cuisine_df
):

    restaurant_cuisines = set(
        restaurant_cuisine_df["Rcuisine"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    user_cuisines = set(
        user_cuisine_df["Rcuisine"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    matching_cuisines = (
        restaurant_cuisines
        & user_cuisines
    )

    return {

        "restaurant_cuisine_types": len(
            restaurant_cuisines
        ),

        "user_cuisine_types": len(
            user_cuisines
        ),

        "matching_cuisine_types": len(
            matching_cuisines
        ),

        "unmatched_user_cuisines": sorted(
            user_cuisines
            - restaurant_cuisines
        ),

        "unmatched_restaurant_cuisines": sorted(
            restaurant_cuisines
            - user_cuisines
        )
    }


# ------------------------------------------------------------
# RUN ALL
# ------------------------------------------------------------

def run_all():

    restaurant_df = load_csv(
        [
            "placeID",
            "latitude",
            "longitude",
            "name"
        ]
    )

    restaurant_cuisine_df = load_csv(
        [
            "placeID",
            "Rcuisine"
        ]
    )

    restaurant_payment_df = load_csv(
        [
            "placeID",
            "Rpayment"
        ]
    )

    hours_df = load_csv(
        [
            "placeID",
            "hours",
            "days"
        ]
    )

    parking_df = load_csv(
        [
            "placeID",
            "parking_lot"
        ]
    )

    user_cuisine_df = load_csv(
        [
            "userID",
            "Rcuisine"
        ]
    )

    user_payment_df = load_csv(
        [
            "userID",
            "Upayment"
        ]
    )

    results = {

        "restaurant":
            restaurant_audit(
                restaurant_df
            ),

        "restaurant_cuisine":
            cuisine_audit(
                restaurant_cuisine_df,
                "placeID",
                "Rcuisine"
            ),

        "restaurant_payment":
            payment_audit(
                restaurant_payment_df,
                "placeID",
                "Rpayment"
            ),

        "hours":
            hours_audit(
                hours_df
            ),

        "parking":
            parking_audit(
                parking_df
            ),

        "user_cuisine":
            cuisine_audit(
                user_cuisine_df,
                "userID",
                "Rcuisine"
            ),

        "user_payment":
            payment_audit(
                user_payment_df,
                "userID",
                "Upayment"
            ),

        "user_ids":
            user_id_audit(
                user_cuisine_df
            ),

        "cross_dataset":
            cross_dataset_audit(
                restaurant_df,
                restaurant_cuisine_df,
                restaurant_payment_df,
                hours_df,
                parking_df,
                user_cuisine_df,
                user_payment_df
            ),

        "recommender_system":
            recommender_audit(
                restaurant_cuisine_df,
                user_cuisine_df
            )
    }

    return results


# ------------------------------------------------------------
# MAIN PROGRAM
# ------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print(
        "RESTAURANT & CONSUMER DATA / "
        "RECOMMENDER SYSTEMS"
    )
    print(
        "REPRODUCIBLE DATA-INTEGRITY AUDIT"
    )
    print("=" * 70)

    try:

        results = run_all()

        for section, result in results.items():

            print(
                f"\n[{section.upper()}]"
            )

            print("-" * 70)

            for key, value in result.items():

                print(
                    f"{key}: {value}"
                )

        print("\n" + "=" * 70)
        print("AUDIT COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except FileNotFoundError as error:

        print("\nERROR:")
        print(error)

        print(
            "\nPlease make sure all 7 CSV files "
            "are in the same folder as audit.py."
        )

    except Exception as error:

        print("\nERROR:")
        print(error)