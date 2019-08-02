1. **`id`**: Identifier for the client
2. **`pol_bonus`**: The bonus system is compulsory in France, but we will only use it here as a possible feature. The coefficient is attached to the driver. It starts at 1 for young drivers (i.e. first year of insurance). Then, every year without a claim, the bonus decreases by 5% until it reaches its minimum of 0.5. Without any claims, the bonus evolution would then be 1 -> 0.95 -> 0.9 -> 0.85 -> 0.8 -> 0.76 -> 0.72 -> 0.68 -> 0.64 -> 0.6 -> 0.57 -> 0.54 -> 0.51 -> 0.5. Every time the driver causes a claim (only certain types of claims are taken into account), the coefficient increases  by  25%,  with  a  maximum  of  3.5. Thus,  the  range  of **`pol_bonus`** is 0.5  to  3.5.
3. **`pol_coverage`**: The coverage are of 4 types `Mini`, `Median1`, `Median2` and, `Maxi`, in this order. As you can guess, `Mini` policies cover only Third Party Liability claims, whereas `Maxi` policies covers all claims, including Damage, Theft, Windshield Breaking, Assistance, etc.
4. **`pol_duration`**: Policy duration represents how old the policy is (how long the customer has renewed it for). It is expressed in years, counted from the beginning of the current year. 
5. **`pol_sit_duration`**: This represents how old the current policy characteristics are. It can be different from **`pol_duration`**, because the same insurance policy could have evolved in the past (e.g. by changing coverage, or vehicle, or drivers etc).
6. **`pol_pay_freq`**: The price of the insurance coverage can be paid annually, bi-annually, quarterly or monthly. Be aware that you must provide a yearly quotation in your answer to the pricing game.
7. **`pol_payd`**: Is the policy under [Pay-A-You-Drive](https://en.wikipedia.org/wiki/Usage-based_insurance); a mileage based policy.
8. **`pol_usage`**: This describes what usage the driver makes from the vehicle, most of time. There are 4 possible values: `WorkPrivate` which is the most common, `Retired` which is presumed to be aimed at retired people (who also are presumed to drive shorter distances), `Professional` which denotes professional usage of the vehicle, and `AllTrips` which is quite similar to Professional (including professional tours). 
9. **`insee_code`**: Essentially a town postcode. This is a  5-digits alphanumeric code used by the French National Institute for Statistics and Economic Studies (hence INSEE) to identify communes and departments in France. There are about 36,000 ‘communes’ in France, but not every one of them is present in the dataset (there are only 18,000 of them). The first 2 digits of insee codes identify the ‘department’ (they are 96 in total, not including overseas departments).
10. **`town_mean_altitude`**: The mean altitude of the town of residence. 
11. **`town_surface_are`**: The approximate physical size of the town. 
12. **`population`**: A number indicating the population of the town. Note that this number is not the population itself, but only represents the population size. 
13. **`lat_long_town`**: The latitude and longitude of the centre of town. 
14. **`commune_code`**: A three digit code specifying the commune. 
15. **`canton_code`**: A 2 digit code specifying the canton.
16. **`city_district_code`**: These are codes specifying information about the borough or district of the town of residence. 
17. **`regional_department_code`**: This is a code specifying the French region.
18. **`drv_drv2`**: The this boolean variable (Yes/No) identifies the presence of a secondary driver on the vehicle. There is always a first driver, for whom characteristics (age, sex, licence) are provided, but a secondary driver is optional, and is present about 30% of the time.
19. **`drv_age1`**: The age of the primary driver
20. **`drv_age2`**: The age of the secondary driver
21. **`drv_sex1`**: European rules force insurers to charge the same price for women and men. But driver’s gender can still be used in academic studies, and that’s why **`drv_sex1`** is still available in the datasets, and can be used as discriminatory variable in this pricing game.
22. **`drv_sex2`**: Gender of the second driver.
23. **`drv_age_lic1`**: **`drv_age_lic1`** is the age of the first driver’s driving licence. As for the other ages, it is expressed in integer years from the beginning of the current year
24. **`drv_age_lic2`**: **`drv_age_lic2`** is the age of the second driver’s driving licence. Be cautious that there are some outliers in the dataset.
25. **`vh_age`**: This variable is the vehicle’s age, the difference between the year of release and the current year. One can consider that **`vh_age`** of 1 or 2 correspond to new vehicles.
26. **`vh_cyl`**: The engine cylinder displacement is expressed  in millilitres in a continuous scale. This variable should be highly correlated with **`vh_din`** of the vehicle which is a proxy for vehicle power.
27. **`vh_din`**: This is a representation of the motor power. Don’t be surprised to find correlations between **`vh_din`**, **`vh_cyl`**, **`vh_speed`** and even value of the vehicle.
28. **`vh_fuel`**: This the fuel type. There are mainly two values `Diesel` and `Gasoline`. A few hybrid vehicles can also be found.
29. **`vh_make`**: The make (brand) of the vehicle. As the database is built from a French insurance company, the three major brands are `Renault`, `Peugeot` and, `Citroën`
30. **`vh_model`**: As a subdivision of the make, vehicle is identified by its model name. The are about 100 different make names in the dataset, and about 1,000 different models. Should you use them, you could consider concatenating **`vh_make`** and **`vh_model`**.
31. **`vh_sale_begin`**: This variable and the next variable are the number of years from the beginning of the current year, from the beginning of the official sale of the vehicle. For example if a car was made available for sale by it's manufacturer 10 years ago, the value for this variable would be 10. This could for instance identify policies that covers very new vehicles or second-hand ones.
32. **`vh_sale_end`**: Identifies the number of years since the vehicle has not been sold officially. Years since sale discontinued. 
33. **`vh_speed`**: This is the maximum speed of the vehicle, as stated by the manufacturer.
34. **`vh_type`**: This can be either `Tourism` or `Commercial`. You’ll  find  more `Commercial` types for `Professional` policy usage than for `WorkPrivate`.
35. **`vh_value`**: The vehicle’s value (replacement value) is expressed in euros, without inflation.
36. **`vh_weight`**: This is the weight (in kg) of the vehicle.
37. **`claim_amount`**: This is essentially what you must predict. These are individual claim amounts range from (approx.) -2,000 to +300,000. 
