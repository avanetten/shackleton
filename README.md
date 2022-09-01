![Alt text](/docs/imgs/header.png?raw=true "")

# Shackleton

The Shackleton codebase leverages remote sensing imagery and machine learning techniques to provide insights into various transportation and evacuation scenarios in an interactive dashboard that conducts real-time computation.  This project extends the [Diet Hadrade](https://github.com/Geodesic-Labs/diet_hadrade) repository that provided static outputs, rather than Shackleton's ability to explore the data and predictions at will. 

Shackleton provides a number of graph theory analytics that combine road graphs (extracted via [CRESI](https://github.com/avanetten/cresi)) with vehicle detections (via [YOLTv5](https://github.com/avanetten/yoltv5), which adapts [YOLOv5](https://github.com/ultralytics/yolov5) to handle satellite imagery) extracted from satellite imagery with advanced computer vision techniques.  This allows congestion to be estimated, as well as optimal lines of communication and evacuation scenarios.  We build these analystics into a [Bokeh](https://docs.bokeh.org/en/latest/) application, which permits exploration of the data and real-time computation of various scenarios:

- Real-time road network status
- Vehicle localization and classification
- Optimized bulk evacuation or ingress
- Congestion estimation and rerouting
- Critical communications/logistics nodes
- Inferred risk from hostile actors

-----

# Quickstart

## 0. Environment

Build the shackleton conda environment by running the following in a terminal:

	shackleton_dir=/Users/ave/projects/GitHub/shackleton
	cd $shackleton_dir
	# conda remove --name shackleton --all
	conda env create --file environment.yml
	conda activate shackleton

## 1. Data

We include a deconstructed SpaceNet 5 sample image (splitting the image is necessary to upload to GitHub) in the [data/test_imagery](data/test_imagery) directory.  Simply recombine the image portions according to [split_recombine_imagery.txt](data/test_imagery/split_recombine_imagery.txt):

	cd $shackleton_dir/data/test_imagery
	cat test_image_split/* > test1_realcog_clip.tif

Alternately, create this image from scratch via [data_curation+inference.ipynb](notebooks/data_curation+inference.ipynb)

## 2. Computer Vision Predictions

In the [results](results/test1) folder we've included both CRESI and YOLTv5 predictions for our sample image, so we can simply use these results for the dashboard (next section).  To run these deep learning models from scratch, see [data_curation+inference.ipynb](notebooks/data_curation+inference.ipynb)

## 3. Execute Shackleton

The Shackleton Dashboard creates a [Bokeh](https://docs.bokeh.org/en/latest/) server that displays the data and connects back to underlying python libraries (such as [NetworkX](https://networkx.org), [OSMnx](https://osmnx.readthedocs.io/en/stable/), and [scikit-learn](https://scikit-learn.org/stable/)).  We also spin up a tile server courtesy of [localtileserver](https://github.com/banesullivan/localtileserver) to visualize our satellite imagery.  
 
Simply execute the following command in the _shackleton_ conda environment to fire up the dashboard:

	cd $shackleton_dir
	test_im_dir=$shackleton_dir/data/test_imagery
	bokeh serve --show src/shackleton/shackleton_dashboard.py --args $test_im_dir/test1_realcog_clip.tif  results/test1/yoltv5/geojsons_geo_0p3/test1_cog_clip_3857.geojson results/test1/cresi/graphs_speed/test1_cog_clip_3857.gpickle

This will invoke the interactive dashboard, which will look something like the image below:
![Alt text](/docs/imgs/dash1.png?raw=true "")


-----

# Step-by-Step Instructions

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


## 2. Road/Vehicle Inference

Open up SageMaker StudioLab and execute the following lines. Execution takes ~3 minutes on the SageMaker GPU.  

### yoltv5

    cd /home/studio-lab-user/shackleton/src/yoltv5/yoltv5
	YAML=/home/studio-lab-user/shackleton/cfg/yoltv5_8class_test_studio_lab.yaml
    time ./test.sh $YAML

### cresi

    cd /home/studio-lab-user/shackleton/src/cresi/cresi
    JSON=/home/studio-lab-user/shackleton/cfg/cresi_8class_test_studio_lab.json
    time ./test.sh $JSON

Since StudioLab will not support Bokeh servers, copy the imagery and road/vehicle predictions back locally in order to run the dashboard.

## 3. Shackleton Dashboard

Build the environment:

	shackleton_dir=/Users/ave/projects/GitHub/shackleton
	cd $shackleton_dir
	# conda remove --name shackleton --all
	conda env create --file environment.yml
	conda activate shackleton

Execute the dashboard:

	cd $shackleton_dir
	test_im_dir=$shackleton_dir/data/test_imagery
	bokeh serve --show src/shackleton/shackleton_dashboard.py --args $test_im_dir/test1_realcog_clip.tif  results/test1/yoltv5/geojsons_geo_0p3/test1_cog_clip_3857.geojson results/test1/cresi/graphs_speed/test1_cog_clip_3857.gpickle


-----

## Further Notes on Motivation

In a disaster scenario where communications are unreliable, overhead imagery often provides the first glimpse into what is happening on the ground, so analytics with such imagery can prove very valuable.  Specifically, the rapid extraction of both vehicles and road networks from overhead imagery allows a host of interesting problems to be tackled, such as congestion mitigation, optimized logistics, evacuation routing, etc.  

A reliable proxy for human population density is critical for effective response to natural disasters and humanitarian crises. Automobiles provide such a proxy. People tend to stay near their cars, so knowledge of where cars are located in real-time provides value in disaster response scenarios. In this project, we deploy the [YOLTv5](https://github.com/avanetten/yoltv5) codebase to rapidly identify and geolocate vehicles over large areas.  Geolocations of all the vehicles in an area allow responders to prioritize response areas.

Yet vehicle detections really come into their own when combined with road network data.  We use the [CRESI](https://github.com/avanetten/cresi) framework to extract road networks with travel time estimates, thus permitting optimized routing.  The CRESI codebase is able to extract roads with only imagery, so flooded areas or obstructed roadways will sever the CRESI road graph; this is crucial for post-disaster scenarios where existing road maps may be out of date and the route suggested by navigation services may be impassable or hazardous.  

Placing the detected vehicles on the road graph enables a host of graph theory analytics to be employed (congestion, evacuation, intersection centrality, etc.).  Of particular note, the test city selected below (Dar Es Salaam) is not represented in any of the training data for either CRESI or YOLTv5.  The implication is that this methodology is quite robust and can be applied immediately to unseen geographies whenever a new need may arise.
