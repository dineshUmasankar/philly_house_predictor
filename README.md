# About

This project is utilizing the [City of Philadelphia Property Assessments Database](https://phila.gov/property/data) in order to predict the market_value of houses by building a machine learning model. The goal of this project is to be able to reliably predict the market_value of the common home in Philadelphia.

# Data Collection

I collected the data from the public database provided by the City of Philadelphia's Office of Property Assessments, by going to their website and downloading the csv. For all intents and purposes of reproducability, my original dataset downloaded from the public database is committed into the repository in the folder `data_collection`. 

# Data Preprocessing

In order to preprocess and clean the data for analysis, I had used an extension in VSCode called Data Wrangler in order to visualize the summary statistics regarding all of the columns.

![Preview of using Data Wrangler on the collected data](report_assets/DataWrangler.png)

### Drop columns with overly missing data
My first cleaning attempt was to drop all columns within the dataset with more than 25% missing values as imputing these values would end up skewing my model. 
As such these were the columns removed:
```
cross_reference
date_exterior_condition
fuel
garage_type
house_extension
mailing_address_1
mailing_address_2
mailing_care_of
market_value_date
number_of_rooms
other_building
separate_utilities
sewer
site_type
suffix
unfinished
unit
utility
```

### Drop columns with high cardinality
There were many columns that were simply identifiers for each entry within the database, and we want our model to find patterns within the data, so as such columns with high cardinality (many distinct values) were identified and removed. At this point of data-cleaning, there were about 500k records. As such, I set a threshold that identifies columns with high cardinality or in other words if a column had 20% unique values (116k), then I would simply drop it.

**NOTE:** I had preserved `lat`,`lng` as I wanted to use the location of each home in hopes of finding a pattern between their location and its price.

These were the columns that were removed from this operation:
```
the_geom
the_geom_webmercator
beginning_point
book_and_page
location
mailing_street
owner_1
owner_2
parcel_number
registry_number
pin
objectid
```

### Filter Properties to Single Family Homes
Since we are building a model to predict the market_value of homes, I decided to remove any properties that are not homes, such as offices and parks, and public property by filtering on the column `category_code_description` for the following values: `SINGLE FAMILY`. This filters the dataset to simply single-family homes, which is what we are trying to predict for.

This reduced the total number of records to 503k which all represent homes in the Philadelphia region.

### Dropping Specific Columns
- `assessment_date` - I do not want to have the model evaluate the market_value based on the assessment date in order to avoid recency bias, where if a house was evaluated more recently, then it could give a slightly higher / lower market value.
- `category_code` - At this point, there is only one category_code, and as such there is no value to having this column.
- `category_code_description` - There remains only single family home(s), so there is no value to having this column, as it only contains 1 value repeated throughout the dataset.
- `year_built_estimate` - This column is a boolean value indicating if the year built was estimated, but I believe that there is more valuable information that can be used towards a valuation such as the actual year that the house was built (`year_built`), and as such, I decided to remove this column.
- `mailing_city_state` & `mailing_zip` - The mailing address of a property doesn't really tell us much about the value of a property, more about whether if it's different than the property adddress, that the owner could potentially be affluent as it could be their secondary property.
- `building_code` - Using the [metadata website](https://metadata.phila.gov/#home/datasetdetails/5543865f20583086178c4ee5/representationdetails/55d624fdad35c7e854cb21a4/?view_287_page=1&view_287_search=basements) from the Office of Philadelphia Property Assessments (OPA), and other external resources, I was still unable to find a clear definition of what all of the codes used within here had meant. There was no updated manual or guide regarding these codes, so I had to drop this column due to ambiguity.
- `street_code` - This is a five-digit number issued by the Water Department, which appears to primarily be used for identification reasons, and as such provides no value towards the valuation of a property's market value.
- `street_direction` - Had too many missing values and unknowns to impute (63%). Also, this is not an ordinal value, it is simply nominal and provides very few value and I do not want these values causing correlation with the market value as there are many other factors that should come above first such as zipcode or lat/lng (indiciating area based correlation to target variable: `market_value`).
- `building_code_description_new` - Nominal attribute that doesn't appear to have much clarification in the OPA's metadata.
- `building_code_new` - Column of two-digit numbers with no representation to what they mean. Even the OPA's metadata didn't have this property listed, so there was no description.
- `building_code_description` - Removed after filtering for misfiled vacant land properties, at this point we have reliably captured most of the single-family homes within the dataset.
- `central_air` - Had 38% missing values and became impossible to impute as it is a binary column that couldn't rely on any other attributes in order to infer.
- `off_street_open` - Metadata did not provide a clear definition regarding this attribute so I found it hard to use as a feature in relation to the target variable.
- `state_code` - It was a redundant datafiled that had 100% PA values at this point of the cleanup, and we only had houses within the philadelphia region
- `house_number` - This number has no meaningful correlation to market_value and is generated at random within the local neighborhood of each property and each street that the property itself is located at.

### Imputations
- `type_heater` - Imputed the Missing Values to H, as this letter represents Undetermined for the type of heating system a property has.
- `basements` - Imputed the Missing Values to K, which is a new definition I created for Unknown, as this metric is an ordinal scales that goes like this:
	```
    0. None – Indicates no basement.
    A. Full Finished – Occupies the entire area under the first floor.
    B. Full Semi-Finished – Could have some finish to include a floor covering,
    and ceiling. It looks more like a living area rather than a basement.
    C. Full Unfinished – Is a typical basement with unfinished concrete floor,
    either rubble stone or cement over stone or concrete walls and would have
    exposed wood joist ceilings.
    D. Full – Unknown Finish
    E. Partial Finished – Occupies a portion under the first floor. Be careful of
    areas under sheds and porches. If there is a garage at basement level then it is a
    partial basement.
    F. Partial Semi-Finished – One or more finished areas.
    G. Partial Unfinished
    H. Partial - Unknown Finish
    I. Unknown Size - Finished
    J. Unknown Size - Unfinished
    K. Unknown - Unknown (NEW DEFINITION CREATED)
    ```
- `quality_grade` - Imputed this value to C, as this represents the average grade for properties within Philadelphia (identified via Data Wrangler).
- `topography` - Based on the metadata provided the philadelphia, most lots are on the street level by average, and so I'm imputing it with F.
    ```
    Most lots in the City are at street level. This is a site that would be at street or sidewalk
    grade or level with a slight contour to permit drainage away from the property. This is
    typical and should be indicated as „F‟ or level. Use one of the following that is most
    appropriate.
    a. Above Street Level - This would be topography where you would have to walk up over
    two (2) flights of steps from the front and rear or is hilly or slopes upward sharply.
    This could pose a problem for development.
    b. Below Street Level - This relates to topography that is below the level of the sidewalk
    and street. You have to go down steps or an embankment. This could pose a problem
    for drainage and development.
    c. Flood Plain - This is a site that falls within an identified Zone A flood hazard zone. This
    is normally found in close proximity to flowing water or a high water table. Typically it
    can be found in South and S.W. Phila., and along the rivers, streams, creeks, etc. It
    could include wetlands or land under water.
    d. Rocky - This relates to areas of the City that have very rocky soil or sub-soil conditions that could have an adverse effect on site grading, construction or installation of sewers
    and water mains. This is normally found in Philly areas, such as N.W. Philadelphia.
    e. Relates to anything not identified here that may be observed that may have some effect
    on value. Indicate what it is in the comments section of this form.
    f. Level.
    ```
### Filter Rows based on specific column values
- `building_code_description` - Removed records that were vacant land properties misfiled under single family homes by filtering and dropping the columns
- `sale_price` - Filter records that sold for a price above $1 as these were entered as placeholder values
- `exempt_building` - This represents if the OPA had exempt this building from certification, and that means if there's a value in this column for a record, that it was exempt, and as such I dropped all the records that were exempt as this means they were not properly assessed in terms of their features.
- `exempt_land` - This represents if the OPA had exempt this land from certification, and that means if there's a value in this column for a record, that it was exempt, and as such I dropped all the records that were exempt as this means they were not properly assessed in terms of their features.
# Feature Engineering
