import pandas as pd

def drop_high_missing_percent_columns(df):
    # Drop columns with more than 25% missing values
    df = df.dropna(thresh=len(df) * 0.25, axis=1)
    return df

def drop_high_cardinality_columns(df):
    # Drop columns with too many distinct values as we want to form clusters within the dataset
    # and create relationships between the features. These distinct values within a single column for
    # every single record blurs the relationships, and as such we are removing it.
    # The entire dataset has about 580k rows at this point, so we are targeting columns with 20% distinct values (116k distinct values)
    high_cardinality_columns = [col for col in df.columns if df[col].nunique() > 116_000]
    # The high_cardinality columns are: ['the_geom', 'the_geom_webmercator', 'book_and_page', 'location', 'parcel_number', 'registry_number', 'pin', 'objectid', 'lat', 'lng']
    # We want to preserve lat,lng so we remove it from the list
    high_cardinality_columns.remove('lat')
    high_cardinality_columns.remove('lng')
    df = df.drop(columns=high_cardinality_columns)
    return df

def filter_single_multifamily_homes(df):
    df = df[(df['category_code_description'] == "SINGLE FAMILY") | (df['category_code_description'] == "MULTI FAMILY")]
    return df

# Load original_dataset from Office of Property Assessments
df = pd.read_csv('original_dataset.csv')

df_clean_missing = drop_high_missing_percent_columns(df.copy())
df_clean_high_cardinality = drop_high_cardinality_columns(df_clean_missing.copy())
df_filter_single_multifamily_homes = filter_single_multifamily_homes(df_clean_high_cardinality.copy())

df_clean_high_cardinality.head()