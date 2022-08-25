![Alt text](/docs/imgs/header.png?raw=true "")

# Shackleton

The Shackleton codebase leverages remote sensing imagery and machine learning techniques to provide insights into various transportation and evacuation scenarios in an interactive dashboard that conducts real-time computation.  This project extends the [Diet Hadrade](https://github.com/Geodesic-Labs/diet_hadrade) repository that provided static outputs, rather than Shackleton's ability to explore the data and predictions at will. 

Shackleton provides a number of graph theory analytics that combine road graphs with vehicle detections extracted from satellite imagery with advanced computer vision techniques.  This allows congestion to be estimated, as well as optimal lines of communication and evacuation scenarios.  We build these analystics into a [Bokeh](https://docs.bokeh.org/en/latest/) application, which permits exploration of the data and real-time computation of various scenarios:

- Real-time road network status
- Vehicle localization and classification
- Optimized bulk evacuation or ingress
- Congestion estimation and rerouting
- Critical communications/logistics nodes
- Inferred risk from hostile actors


-----
## 0. Create the SageMaker StudioLab environment

GPU inference of roads and vehicles can be accomplished in the freely available Amazon SageMaker StudioLab.  

    # install yolov5
    # https://github.com/ultralytics/yolov5
    cd /home/studio-lab-user
    conda activate default
    git clone https://github.com/ultralytics/yolov5
    cd yolov5
    pip install -r requirements.txt  # install

    # update with geo packages
    conda install -c conda-forge gdal
    conda install -c conda-forge osmnx
    conda install -c conda-forge osmnx=0.12 
    conda install -c conda-forge scikit-image
    conda install -c conda-forge statsmodels
    conda install -c conda-forge matplotlib
    conda install -c conda-forge ipykernel 
    pip install torchsummary
    pip install utm
    pip install numba
    pip install jinja2==2.10
    pip install geopandas==0.8
    
    # clone shackleton codebase
    git clone https://github.com/avanetten/shackleton.git

    # clone YOLTv5 and CRESI
    cd /home/studio-lab-user/shackleton/src/
    git clone https://github.com/avanetten/cresi.git
    git clone https://github.com/avanetten/yoltv5.git

## 1. Data Prepatation 

See [data_curation+inference.ipynb](notebooks/data_curation+inference.ipynb).

-----
## 2. Road/Vehicle Inference

Open up SageMaker and execute the following lines. Execution takes ~3 minutes on the SageMaker GPU.

### yoltv5

    cd /home/studio-lab-user/shackleton/src/yoltv5/yoltv5
	YAML=/home/studio-lab-user/shackleton/cfg/yoltv5_8class_test_studio_lab.yaml
    time ./test.sh $YAML

### cresi

    cd /home/studio-lab-user/shackleton/src/cresi/cresi
    JSON=/home/studio-lab-user/shackleton/cfg/cresi_8class_test_studio_lab.json
    time ./test.sh $JSON


-----

## 3. Shackleton Dashboard

	cd /Users/ave/projects/geodesic/shackleton
	# conda remove --name shackleton --all
	conda env create --file environment.yml
	conda activate shackleton

	# # Optional: Ensure the image is a valid COG:
	# test_im_dir=/Users/ave/projects/GitHub/shackleton/data/private/test_imagery/input
	# cd $test_im_dir
	# gdal_translate test1_cog_clip.tif test1_realcog_clip.tif -of COG -co COMPRESS=LZW	

	shackleton_dir=/Users/ave/projects/GitHub/shackleton
	test_im_dir=$shackleton_dir/data/private/test_imagery/input
	conda activate shackleton
	bokeh serve --show src/shackleton/shackleton_dashboard.py --args $test_im_dir/test1_realcog_clip.tif  results/test1/yoltv5/geojsons_geo_0p3/test1_cog_clip_3857.geojson results/test1/cresi/graphs_speed/test1_cog_clip_3857.gpickle

This will invoke the interactive dashboard, which will look something like the image below:

![Alt text](/docs/imgs/dash1.png?raw=true "")


-----

### Further Notes on Motivation

In a disaster scenario where communications are unreliable, overhead imagery often provides the first glimpse into what is happening on the ground, so analytics with such imagery can prove very valuable.  Specifically, the rapid extraction of both vehicles and road networks from overhead imagery allows a host of interesting problems to be tackled, such as congestion mitigation, optimized logistics, evacuation routing, etc.  

A reliable proxy for human population density is critical for effective response to natural disasters and humanitarian crises. Automobiles provide such a proxy. People tend to stay near their cars, so knowledge of where cars are located in real-time provides value in disaster response scenarios. In this project, we deploy the [YOLTv5](https://github.com/avanetten/yoltv5) codebase to rapidly identify and geolocate vehicles over large areas.  Geolocations of all the vehicles in an area allow responders to prioritize response areas.

Yet vehicle detections really come into their own when combined with road network data.  We use the [CRESI](https://github.com/avanetten/cresi) framework to extract road networks with travel time estimates, thus permitting optimized routing.  The CRESI codebase is able to extract roads with only imagery, so flooded areas or obstructed roadways will sever the CRESI road graph; this is crucial for post-disaster scenarios where existing road maps may be out of date and the route suggested by navigation services may be impassable or hazardous.  

Placing the detected vehicles on the road graph enables a host of graph theory analytics to be employed (congestion, evacuation, intersection centrality, etc.).  Of particular note, the test city selected below (Dar Es Salaam) is not represented in any of the training data for either CRESI or YOLTv5.  The implication is that this methodology is quite robust and can be applied immediately to unseen geographies whenever a new need may arise.



