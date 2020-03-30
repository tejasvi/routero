## ℹ️ Project information

**Short Project Description:** AI-based route optimization and visualization tool for sales vehicles.

 _Optimization Mode_: Maximize the total Potential Sales Delivered with constraints in inputs in minimum cost
_Recommendation Mode_: Recommend the Location of Distribution Points and/or Number of Vehicles, Types of vehicles at the distribution Point to cover X% of Potential sales and best cost for that coverage.
_Visualization_: The tool is supported by a visualization in Streamlit and Mapbox which shows the delivery Points, distribution Points and the calculated route for each delivery vehicle.

**Team Name:** Genesis

**Team Members:** Tejasvi S Tomar @tejasvi, Pooja Dhane @PoojaDh

**Data used:** Synthetic data based on [The heterogeneous fleet vehicle routing problem with overloads and time windows]( https://www.sciencedirect.com/science/article/pii/S0925527313000388) after heavy modification according to the given specifications.

**Azure Services Used:**
* Azure Maps (distance API and routing)
* Azure Machine Learning (training modified K-NN)
* Azure Notebooks (prototyping and data analysis)
* Azure App Service (Web-based deployment using MapBox and Streamlit)


## 🔥 Hackathon Pitch

In the fast-developing logistics and supply chain management fields, one of the critical problems in the decision-making system is that how to arrange a proper supply chain for a lot of destinations and suppliers and produce a detailed supply schedule under a set of constraints. Solutions to the multisource vehicle routing problem (MDVRP) help in solving this problem in case of transportation applications.

Given the locations of sources and destinations, the MDVRP requires the assignment of destination to sources and the vehicle routing for visiting them. Each vehicle originates from one source, serves the destinations assigned to that source, and returns to the same source. The objective of the MDVRP is to serve all destinations while minimizing the total travel distance (hence cost) under the constraint that the total demands of served destinations cannot exceed the capacity of the vehicle for each route.

This project uses a heuristic algorithm to solve this problem. The proposed algorithm consists of two phases:

* Phase 1: Finds the centroids which will be delivered by each vehicle by using a modified k-means clustering algorithm.
* Phase 2: Choose the destinations for each centroid (i.e., source points)
* Phase 3: Assign the vehicle to the subsets of the destination points using K-means
* Phase 4: Optimize the vehicle routing (Using _Travelling Salesman_ optimization)

## 🔦 Other highlights

The hackathon was a great learning opportunity to get familiar with cloud-based development. Since we were free to choose our dataset, a considerable effort put into ensuring our synthetic data was sufficiently realistic. Apart from that, we experimented with multiple existing approaches to VRP, including genetic algorithms and recursive-DBSCAN. However, we found our strategy to be most performant when scaled industrial level. One of the primary reasons is the relative simplicity and the speed of execution, which allows much-needed flexibility. 


**Demo Link:** http://routero,herokuapp.com

**Presentation Link:** https://1drv.ms/p/s!AlbLbaPx_OoCnAC3pbRIdt6unRrF?e=Z2dhSu
