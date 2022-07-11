# The Effect of Exchange Rate Volatility on Malaysia External Trade
## Abstract
This project analysed the effects of exchange rate volatility on export and import of Malaysia by employing monthly trade data for the period from January 2010 to December 2020. The project is extended to cover both sectoral and country specific export and import volumes. The major aim of this project is to show how fluctuations in foreign exchange rate change the volume of exports and imports among various commodities in Malaysia. In this project, 6 different regression algorithms are utilized to explain the effect of exchange rate volatility on export and import in Malaysia. The impact of features on the target feature is analysed using linear, ridge, lasso, elastic net, support vector machine and random forest regression algorithms. Based on the results of these six algorithms for Malaysia, the volatility of exchange rate has significant impact on some commodities.
## Data Collection
The export and import based on SITC 1-digit code is obtained from Malaysia External Trade Statistics (METS) online trade database. The exchange rates are retrieved Central Bank of Malaysia database. The dataset covers trade statistics over the period between January 2010 and December 2020. The export dataset contains more that 279,510 rows with 5 variables while import dataset has more than 171,965 rows with 5 variables. The commodities sections in METS database complies with classification of product group based on OECD standards. As for the exchange rates, only the top eight countries were used in the project. 
| SITC Code | Commodity |
| ------------- | ------------- |
| 0 | Food |
| 1 | Beverages and Tobacco |
| 2 | Crude Materials, Inedible) |
| 3 | Mineral Fuels and Lubricants |
| 4	|Animal and Vegetable Oils and Fats|
| 5	|Chemical |
| 6 |	Manufactured goods|
| 7 |Machinery & Transport Equipment|
| 8 |Miscellaneous Manufactured Articles|
| 9 |	Miscellaneous Transactions and Commodities|

## Methodology
The model is formulated as regression problem. The contribution of explanatory variables is analysed using regression analysis on time series value of trade and foreign exchange rates. This project employed correlation analysis and six regression algorithms: Linear, Lasso, Ridge, and Elastic Net, Support Vector Machine and Random Forest to analyses the impact of exchange rate volatility on the change of export and import in Malaysia on commodities basis. After data pre-processing, variable selection and feature engineering, the final dataset was aggregated by month and year and contained 132 rows and 36 variables. New variables were feature engineered were total export, total import, trade balance, growth rate export and growth rate import.

## Results and Analysis
1. The Performance of Malaysia’s Trade In 10 Years
Over the last 10 years, there was ups and downs in Malaysia’s trade performance based on Chart 1. After a recovery in international trade in 2017, global economic conditions started deteriorating in the second half of 2018 and further in 2019 due to the trade tensions between United States of American and China and a negative global output outlook generally. As observed in Chart 2, the trade downturn of 2019 has been widespread across all geographic regions including Malaysia. The Malaysia’s growth rate for export and import took a sharp fall during that crisis period and further drop in 2020 due to the COVID-19 pandemic. After the reopening of borders and business, the trade bounced back in 2021 as seen in Chart 2.
![image](https://user-images.githubusercontent.com/58675575/178284791-347b472e-9921-44be-8292-1610f257f9be.png)
![image](https://user-images.githubusercontent.com/58675575/178284860-94fc5e2d-5463-4df8-9ce8-829c1183e098.png)
2. Major Trading Partners
Singapore (13.95%), China (13.69%), United States (9.55%), Japan (8.65%) and Hong Kong (5.55%) remained the top five major countries of destination for over 10 years to Malaysia’s exports while for import, the top five major countries of origin for Malaysia were China (18.64%), Singapore, Japan, United States and Taiwan based on Chart 3. As observed in Chart 4, Singapore, Hong Kong, United States, India and Japan were the top five major countries that contributed the most to Malaysia’s trade.
![image](https://user-images.githubusercontent.com/58675575/178285049-aa9c85a9-956b-4be1-928e-d7ff7d372e2a.png)
![image](https://user-images.githubusercontent.com/58675575/178285149-17eb282e-528d-4ab5-9b12-64d10fa2d05c.png)
3. Major and Selected Commodities
There are ten commodities sections based on SITC 1-digit code. The top five major export commodities for the past 10 years were machinery and transport equipment (41.84%), mineral fuels, lubricants etc (16.19%), miscellaneous manufactured articles (11.36%), manufactured goods (9.28%) and chemicals (7.72%) while for import, the top five major commodities were machinery and transport equipment (43.69%), mineral, fuels, lubricants, etc (13.22%), manufactured goods (12.14%), chemicals (10.19%) and miscellaneous manufactured articles (6.72%). For over 10 years, machinery and transport equipment were the largest contributions to Malaysia’s trade where Malaysia imports at 43.69 per cent more than exports which was at 41.84 per cent. However, for mineral and fuels, lubricants etc, Malaysia exports (16.19%) more than imports (13.22%). The other commodities that Malaysia exports more than import were miscellaneous manufactured articles and animals and vegetables oil and fats. As for manufactured goods, chemicals, food, crude materials, inedible and miscellaneous transactions and commodities, Malaysia imports more than exports. Out of 10 SITC 1-digit code, only four commodities Malaysia exports more than imports, the other six commodities were imported more than exported.
![image](https://user-images.githubusercontent.com/58675575/178285383-32e6da82-50bb-4d51-81ad-2c610c2c4e79.png)


