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
**NOTE: ** I had preserved lat,lng as I wanted to use the location of each home in hopes of finding a pattern between their location and its price.

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

### Filter Properties to Single Family / Multi Family Homes
Since we are building a model to predict the market_value of homes, I decided to remove any properties that are not homes, such as offices and parks, and public property by filtering on the column `category_code_description` for the following values: `SINGLE FAMILY` or `MULTI FAMILY`. This filters the dataset to simply single-family homes and townhouses and duplexes, which is what we are trying to predict for.

This reduced the total number of records to 503k which all represent homes in the Philadelphia region.