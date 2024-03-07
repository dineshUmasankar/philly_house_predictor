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

**NOTE:** I had dropped lat and lng because they were too precise of a measurement and wanted to rely on zip code instead as it represents a localized region within philadelphia quite well.

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
- `general_construction` - Too many missing values and unclear definitions for what each unique value in this feature represents. Hard to determine an average in order to impute, so I dropped it.
- `quality_grade` - This is also a feature with unclear definitions that has both letters and numerical representation that was not clarified in the metadata. The data is too messy in order to determine any valuable correlations to the target variable.
- `exempt_land` - The correlation coefficient between market_value and exempt land was 0.06 representing a weak correlation and so this column got removed in the idea of dimensionality reduction.
- `sale_price` - After manually, analyzing the first quartile and the outliers of this data manually, I realized that there are too many values here with crazy extremes, and that this column would have to be dropped.
- `sale_date` - After removing sales after the end of 2023, I decided to drop this column in order to avoid recency bias within the model.

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
- `sale_date` - I decided to make the cut-off upto the end of December 2023, as some of these properties listed within the dataset haven't even been fully constructed and so its' market value also tends to have many placeholders.
- `basements` - I removed all values that were not part of the metadata's basement indexing. There were some extraneous values such as 1,2,3,4 that did not have a description.
- `interior_condition` - Removed all values that were not part of the metadata for interior condition, such as 8. I had also dropped 0 from the valid definition criteria as I did not want to include not applicable. As this the definition for interior condition:
    ```
    This could indicate the overall condition of the interior.
    0. Not Applicable.
    2. New / Rehabbed – Noticeably new construction then surrounding properties
    in the GMA. Property is superior to most other properties on the block. Usually
    the following exterior improvements can be observed.

    3. Above Average - would indicate that some work had been done or the level of
    maintenance has been beyond what is typical for the area. The interior would show
    very well and be in move in condition.
    4. Average – would be typical.
    5. Below Average – would be the opposite. It could appear that maintenance has
    been let go and things normally were not repaired or replace on a regular basis.

    6. Vacant – No occupancy. FHA, VA, FNMA signs may be on the property.
    Property has been secured with fresh plywood over doors and windows.
    7. Sealed / Structurally Compromised, Open to the Weather –
    Doors and windows have been covered over by plywood, tin, concrete block or
    stucco. No interior access. Some or no windows, no door or door open, evidence
    of past abuse by vandals such as graffiti, missing railings, deteriorated wood and
    metal, etc. Scorch marks and/or fire and water damage to exterior brick, siding,
    bays, etc. Broken windows with blackened and charred interior.
    ```

### Filter Rows due to missing values
The following rows at this point of data preprocessing have missing values across these features.
![Missing Values By Column shown in graphic](report_assets/MissingValsByCol.png)

For all of these missing values by column, I've decided to drop them instead of going for an imputation as they highly depend on the geographical region, and also rely on the market_value in order to inference properly, but since we are trying to infer market_value, I did not want to influence the dataset creating an influenced dataset.

Moreover, we have 300k records at this point in time, and so I beleive in order to retain the purity of the dataset, it is best if we drop the 13k rows of missing values all together instead of influencing the training data inapproriately.

## Conclusion to Data Preprocessing (Clean Missing Values + Imputation)
Finally, we have a cleaned dataset with categorical, numerical values that total about 330k records. Next, we have to encode certain columns appropriately based on their categorical values (ordinal vs. nominal).

### Categorical Transformations

- `basements`: Using the Ordinal Encoder, I transformed the ordinal values: `['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']`
- `exterior_conditions`: Similar to basements, except these labels are numerical, however they each represent order as the higher the number is, than the better the condition. `[7, 6, 5, 4, 3, 2, 1, 0]`. At this point, after filtering, I did not have any values beyond 7.
    ```
    Relates to how the exterior appears based on observation.
    0. Not Applicable
    1. NEWER CONSTRUCTION – Noticeably newer construction then surrounding properties in the GMA.
    2. REHABILITATED – Property is superior to most other properties on the block.
    Usually the following exterior improvements can be observed:
    New full or partial brick or other material front
    New windows, doors
    New concrete sidewalks, steps, porch, patio or decks
    If you are not certain, use ABOVE AVERAGE.
    3. ABOVE AVERAGE – A well-maintained property where the owner does preventive
    maintenance on an on going basis and reacts to any deferred maintenance as it starts
    to occur. Exterior physical condition is better than average and less than
    4. REHABILITATED.
    5. AVERAGE – This is the typical and most common physical condition observed at the exterior of most properties on the subject block. No significant concrete work, pointing, painting, carpentry or work to trim exterior walls, doors, windows or bay is required. No obvious defects. Majority of properties in the block or GMA are in this condition.
    6. BELOW AVERAGE – Excessive deferred maintenance, wear and tear, abuse, and/or minor vandalism, or unrepaired minor fire damage. These items are starting to add up and take their toll.
    7. VACANT – No occupancy. FHA, VA, FNMA signs may be on the property. Property has been secured with fresh plywood over doors and windows.
    8. SEALED – Doors and windows have been covered over by plywood, tin, concrete block or stucco. No interior access.
    9. STRUCTURALLY COMPROMISED, OPEN TO THE WEATHER - Some or no windows, no door or door open, evidence of past abuse by vandals such as graffiti, missing railings, deteriorated wood and metal, etc. Scorch marks and/or fire and water damage to exterior brick, siding, bays, etc. Broken windows with blackened and charred interior.
    ```
- `type_heater`: Similar to the rest, ordinally encoded based on the following order as described by the OPA:
    ```
    Type of heater or heating system.
    A. Hot air (ducts)
    B. Hot water (radiators or baseboards)
    C. Electric baseboard
    D. Heat pump (outside unit).
    E. Other
    G. Radiant
    H. Undetermined / None
    ```
- `interior_condition`: Similar to rest, Ordinal Encoding after removing values without any clear definitions from the OPA's metadata, encoded in the order as seen below but 0's and 2's were removed.
    ```
    This could indicate the overall condition of the interior.
    0. Not Applicable.
    2. New / Rehabbed – Noticeably new construction then surrounding properties
    in the GMA. Property is superior to most other properties on the block. Usually
    the following exterior improvements can be observed.

    3. Above Average - would indicate that some work had been done or the level of
    maintenance has been beyond what is typical for the area. The interior would show
    very well and be in move in condition.
    4. Average – would be typical.
    5. Below Average – would be the opposite. It could appear that maintenance has
    been let go and things normally were not repaired or replace on a regular basis.

    6. Vacant – No occupancy. FHA, VA, FNMA signs may be on the property.
    Property has been secured with fresh plywood over doors and windows.
    7. Sealed / Structurally Compromised, Open to the Weather –
    Doors and windows have been covered over by plywood, tin, concrete block or
    stucco. No interior access. Some or no windows, no door or door open, evidence
    of past abuse by vandals such as graffiti, missing railings, deteriorated wood and
    metal, etc. Scorch marks and/or fire and water damage to exterior brick, siding,
    bays, etc. Broken windows with blackened and charred interior.
    ```
# Feature Engineering
