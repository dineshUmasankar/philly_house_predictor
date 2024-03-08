# %%
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from category_encoders import BinaryEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

    # Remove records with placeholder values for sale_price and market_value
    df = df[df['sale_price'] > 1]
    df = df[df['market_value'] > 1]
    
    # Have ensured at this point most of the dataset is primarily of single-family homes.
    df = df.drop(columns=['building_code_description'])
    df = df.drop(columns=['building_code_description_new'])
    
    # Removed Central Air as it's a binary variable with no other meaningful reference to impute from (38% missing)
    df = df.drop(columns=['central_air'])

    # Unclear Meaning from Metadata
    df = df.drop(columns=['off_street_open'])

    # Dropping State Code as this information isn't pertinent to relations of the market value within Philadelphia
    df = df.drop(columns=['state_code'])

    # Dropping House Number as these are based more on the local neighborhoods within Philadelphia which tends to have very little value in regards to market_value
    df = df.drop(columns=['house_number'])

    # Nearly 12k missing values and there's unclear definition on what is an average and what each letter represents for general construction from the metadata
    df = df.drop(columns=['general_construction'])

    # Unclear Definitions and too many messy values
    df = df.drop(columns=['quality_grade'])

    # Dropped due to low correlation to market value: 0.06
    df = df.drop(columns=['exempt_land'])

    # Dropped due to too wide spread when manually analyzing
    df = df.drop(columns=['sale_price'])

    return df

def impute_columns(df):
    # Imputed Missing Values in basement column with new attribute K.
    df = df.fillna({'basements': "K"})

    # Imputed Missing Values in type_heated column with attribute H (represents missing/unknown heating type for property).
    df = df.fillna({'type_heater': "H"})

    # Imputed Missing Values in topography column with attribute F (represents street level as most properties in philadelphia are at street level according OPA).
    df = df.fillna({'topography': "F"})

    return df

def drop_missing_vals_records(df):
    df = df.dropna(subset=['census_tract', 'depth', 'exterior_condition', 'fireplaces', 'frontage', 'garage_spaces', 
                           'geographic_ward', 'interior_condition', 'market_value', 'number_of_bathrooms', 'number_of_bedrooms', 
                           'number_stories', 'parcel_shape', 'taxable_building', 'total_area', 'total_livable_area', 'view_type',
                            'year_built', 'zip_code', 'zoning'])

    return df
    
# Load original_dataset from Office of Property Assessments
df = pd.read_csv('original_dataset.csv')

df_clean_missing = drop_high_missing_percent_columns(df.copy())
df_clean_high_cardinality = drop_high_cardinality_columns(df_clean_missing.copy())
df_filter_single_family_homes = filter_single_multifamily_homes(df_clean_high_cardinality.copy())
df_remove_specific = drop_specific(df_filter_single_family_homes.copy())
df_impute_columns = impute_columns(df_remove_specific.copy())
df_remove_missing_vals_records = drop_missing_vals_records(df_impute_columns.copy())

# Define a function to filter dates before 2024
def filter_dates(row):
    year = int(row['sale_date'][:4])
    return year < 2024

# Apply the filter function to the DataFrame and drop the column as we want to avoid recency bias
df_filter_saledate = df_remove_missing_vals_records[df_remove_missing_vals_records.apply(filter_dates, axis=1)].drop(columns=['sale_date'])

def filter_specific(df):
    # Constraining Basements to the following valid indexes and transform to ordinal encoding
    valid_basement = ['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    df = df[df['basements'].isin(valid_basement)]

    # Constraining type_heater to valid definitions from metadata
    valid_type_heater = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    df = df[df['type_heater'].isin(valid_type_heater)]

    # Constraining view_type to valid definitions from metadata
    valid_view_types = ['I', 'H', 'D', 'A', 'C', '0', 'E', 'B']
    df = df[df['view_type'].isin(valid_view_types)]

    # Constraining topography to valid definitions from metadata
    valid_topography_types = ['A', 'B', 'C', 'D', 'E', 'F']
    df = df[df['topography'].isin(valid_topography_types)]
    
    # Constraining parcel_shape to valid definitions from metadata
    valid_parcel_shape = ['A', 'B', 'C', 'D', 'E']
    df = df[df['parcel_shape'].isin(valid_parcel_shape)]
    
    
    return df



df_filter_specific = filter_specific(df_filter_saledate.copy())

# %%
df_encode = df_filter_specific.copy()
# Basement Ordinal Encoding
valid_basements = ['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
basements_encoder = OrdinalEncoder(categories=[valid_basements])
df_encode['basements_encoded'] = basements_encoder.fit_transform(df_encode[['basements']])
df_encode = df_encode.drop(columns=['basements'])

# Exterior Condition Ordinal Encoding
valid_exterior_conditions = sorted(df_encode['exterior_condition'].unique().astype(int))
exterior_encoder = OrdinalEncoder(categories=[valid_exterior_conditions])
df_encode['exterior_encoded'] = exterior_encoder.fit_transform(df_encode[['exterior_condition']])
df_encode = df_encode.drop(columns=['exterior_condition'])

# Interior Condition Ordinal Encoding
valid_interior_conditions = sorted(df_encode['interior_condition'].unique().astype(int))
interior_encoder = OrdinalEncoder(categories=[valid_interior_conditions])
df_encode['interior_encoded'] = interior_encoder.fit_transform(df_encode[['interior_condition']])
# Remove records with invalid interior condition that do not have any definitions within metadata OPA.
df_encode = df_encode[df_encode['interior_encoded'] != 0]
df_encode = df_encode[df_encode['interior_encoded'] != 1]
df_encode = df_encode[df_encode['interior_encoded'] != 8]
df_encode = df_encode.drop(columns=['interior_condition'])

# Type Heater Ordinal Encoding
valid_type_heater = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
heater_encoder = OrdinalEncoder(categories=[valid_type_heater])
df_encode['type_heater_encoded'] = heater_encoder.fit_transform(df_encode[['type_heater']])
df_encode = df_encode.drop(columns=['type_heater'])

# View Type One Hot Encode Encoding
view_encoder = OneHotEncoder(drop=None, sparse_output=False)
view_encoder.fit(df_encode[['view_type']])
df_encode[['view_type_I', 'view_type_H', 'view_type_D', 'view_type_A', 'view_type_C', 'view_type_0', 'view_type_E', 'view_type_B']] = view_encoder.transform(df_encode[['view_type']])
df_encode = df_encode.drop(columns=['view_type'])

# Topograhy One Hot Encode (as there is no order and we don't want to influence priority)
topography_encoder = OneHotEncoder(drop=None, sparse_output=False)
topography_encoder.fit(df_encode[['topography']])
df_encode[['topography_A', 'topography_B', 'topography_C', 'topography_D', 'topography_E', 'topography_F']] = topography_encoder.transform(df_encode[['topography']])
df_encode = df_encode.drop(columns=['topography'])

# Parcel Shape One Hot Encode (as there is no order and we don't want to influence priority)
parcel_shape_encoder = OneHotEncoder(drop=None, sparse_output=False)
parcel_shape_encoder.fit(df_encode[['parcel_shape']])
df_encode[['parcel_shape_A', 'parcel_shape_B', 'parcel_shape_C', 'parcel_shape_D', 'parcel_shape_E']] = parcel_shape_encoder.transform(df_encode[['parcel_shape']])
df_encode = df_encode.drop(columns=['parcel_shape'])

# Homestead Exemption
print(df_encode[['homestead_exemption', 'market_value']].corr())
df_encode['homestead_exemption_encoded'] = df_encode['homestead_exemption'].clip(0, 1)
print(df_encode[['homestead_exemption_encoded', 'market_value']].corr())
df_encode = df_encode.drop(columns=['homestead_exemption'])

# Zoning (35 different zoning, and nominal attribute, we are going to binary encode this column)
# Binary Encoded in the following order: ['RSA5' 'RSA3' 'RM1' 'RMX2' 'RSD3' 'RM4' 'CA1' 'RSA2' 'RSD1' 'CMX4' 'CMX5'
#  'CMX2' 'RM2' 'RSA4' 'RSA1' 'ICMX' 'RM3' 'RMX3' 'RTA1' 'RMX1' 'I3' 'IRMX'
#  'CMX1' 'CMX3' 'RSD2' 'I2' 'RSA6' 'I1' 'CMX2.5' 'SPINS' 'SPPOA'
#  'RSD1|RSD3' 'ICMX|SPPOA' 'CA2' 'RSA5|RSA5']
zoning_encoder = BinaryEncoder(cols=['zoning'])
df_encode = zoning_encoder.fit_transform(df_encode)

# Zipcode Binary Encoding
zipcode_encoder = BinaryEncoder(cols=['zip_code'])
df_encode = zipcode_encoder.fit_transform(df_encode)

# Year Built Binary Encoding
year_built_encoder = BinaryEncoder(cols=['year_built'])
df_encode = year_built_encoder.fit_transform(df_encode)

# Geographic Ward Binary Encoding
year_built_encoder = BinaryEncoder(cols=['geographic_ward'])
df_encode = year_built_encoder.fit_transform(df_encode)

# Census Tract Binary Encoding
year_built_encoder = BinaryEncoder(cols=['census_tract'])
df_encode = year_built_encoder.fit_transform(df_encode)

# Street Name
street_name_encoder = BinaryEncoder(cols=['street_name'])
df_encode = street_name_encoder.fit_transform(df_encode)

# Street Designation
street_name_encoder = BinaryEncoder(cols=['street_designation'])
df_encode = street_name_encoder.fit_transform(df_encode)

# %%
def remove_outliers_winsorize(df, column_name, percentiles=[5, 95]):
    # Winsorize outliers (capping them to specific percentiles)
    capped_column_name = f'{column_name}_capped'
    df[capped_column_name] = df[column_name].clip(
        lower=df[column_name].quantile(percentiles[0]/100),
        upper=df[column_name].quantile(percentiles[1]/100)
    )

    # Calculate IQR using winsorized data
    Q1 = df[capped_column_name].quantile(0.25)
    Q3 = df[capped_column_name].quantile(0.75)
    IQR = Q3 - Q1

    # Define lower and upper bounds for outliers based on winsorized IQR
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    # Filter out outliers using winsorized IQR
    df_filtered = df[
        (df[capped_column_name] >= lower_bound) &
        (df[capped_column_name] <= upper_bound)
    ]

    # Drop the original column
    df_filtered.drop(columns=[column_name], inplace=True)

    return df_filtered

# %%
df_remove_outliers = df_encode.copy()
df_remove_outliers = remove_outliers_winsorize(df_remove_outliers, 'depth')
df_remove_outliers = remove_outliers_winsorize(df_remove_outliers, 'frontage')
df_remove_outliers = remove_outliers_winsorize(df_remove_outliers, 'garage_spaces')
df_remove_outliers = remove_outliers_winsorize(df_remove_outliers, 'total_area')
df_remove_outliers = remove_outliers_winsorize(df_remove_outliers, 'total_livable_area')
df_remove_outliers = remove_outliers_winsorize(df_remove_outliers, 'taxable_building')
df_remove_outliers = remove_outliers_winsorize(df_remove_outliers, 'taxable_land')
df_remove_outliers = remove_outliers_winsorize(df_remove_outliers, 'exempt_building')
df_remove_outliers = remove_outliers_winsorize(df_remove_outliers, 'market_value', percentiles=[5,99])
df_remove_outliers.to_csv('filtered.csv')
df_remove_outliers

# %%
# Identify columns to scale
columns_to_scale = ['fireplaces', 'number_of_bathrooms', 'number_of_bedrooms', 'number_stories',
           'basements_encoded', 'exterior_encoded', 'interior_encoded',
           'type_heater_encoded', 'homestead_exemption_encoded', 'exempt_building_encoded',
           'depth_capped', 'frontage_capped', 'garage_spaces_capped',
           'total_area_capped', 'total_livable_area_capped',
           'taxable_building_capped', 'taxable_land_capped',
           'exempt_building_capped']

# Identify columns not to scale
columns_not_to_scale = [col for col in df_remove_outliers.columns if col not in columns_to_scale]

# Final list of all columns by name
scaled_cols = columns_to_scale + columns_not_to_scale

# Create a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('scaled_features', StandardScaler(), columns_to_scale),
    ],
    remainder='passthrough'  # Keep any columns not specified in transformers
)

# Create a pipeline with the preprocessor and any other steps (e.g., model)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
])

scaled_dataset = pd.DataFrame(pipeline.fit_transform(df_remove_outliers), columns=scaled_cols)
scaled_dataset.to_csv('scaled.csv')