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
    # Filter out only homes (single / multi-family homes)
    df = df[(df['category_code_description'] == "SINGLE FAMILY")]
    return df

def drop_specific(df):
    # Dropped in order to avoid recency bias
    df = df.drop(columns=['assessment_date'])

    # There's only a single category: Single Family Home(s) and both of these columns have redundant information of the same single value representing single-family home(s).
    df = df.drop(columns=['category_code'])
    df = df.drop(columns=['category_code_description'])

    # This column only re-assures us if the year built has been estimated and doesn't really provide us value towards estimating an valuation of the property.
    # More valuable alternative is the year_built column which also has much less missing values.
    df = df.drop(columns=['year_built_estimate'])

    # Dropped in order to avoid recency bias
    df = df.drop(columns=['recording_date'])
    # Drop columns: 'mailing_city_state', 'mailing_zip' b/c the mailing address / state of a property owner doesn't indicate a property's worth.
    df = df.drop(columns=['mailing_city_state', 'mailing_zip'])

    # Dropped building_code because it was not possible to find a clear definition of what the codes had represented.
    # There is no updated code manual nor is it even described within the metadata of the Office of Phildelphia's Assessment 
    df = df.drop(columns=['building_code'])

    # Dropped b/c water department uses this as some form of identification number that was not clarified by the Office of Philadelphia's metadata.
    df = df.drop(columns=['street_code'])

    # Too many missing values and near impossible to impute. It is simply a nominal attribute that is hard to attribute towards our target variable and very susceptible
    # to forming bad patterns within our model such as north direction could increase market valuation, but the zipcode and lat/lng are better indicators of an area based
    # valuation.
    df = df.drop(columns=['street_direction'])
    
    # Unknown definition of two-digit numbers, weren't even listed in the OPA's metadata
    df = df.drop(columns=['building_code_new'])

    # Remove all Vacant Land Properties from the dataset
    df = df[~df['building_code_description'].str.contains("VACANT", regex=False, na=False)]

    df = df.drop(columns=['building_code_description'])
    return df

# Load original_dataset from Office of Property Assessments
df = pd.read_csv('original_dataset.csv')

df_clean_missing = drop_high_missing_percent_columns(df.copy())
df_clean_high_cardinality = drop_high_cardinality_columns(df_clean_missing.copy())
df_filter_single_family_homes = filter_single_multifamily_homes(df_clean_high_cardinality.copy())
df_remove_specific = drop_specific(df_filter_single_family_homes.copy())

df_remove_specific.head()