{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<p style=\"text-align:center\">\n",
    "    <a href=\"https://skills.network\" target=\"_blank\">\n",
    "    <img src=\"https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png\" width=\"200\" alt=\"Skills Network Logo\">\n",
    "    </a>\n",
    "</p>\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Space X  Falcon 9 First Stage Landing Prediction**\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    " ## Lab 2: Data wrangling \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Estimated time needed: **60** minutes\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this lab, we will perform some Exploratory Data Analysis (EDA) to find some patterns in the data and determine what would be the label for training supervised models. \n",
    "\n",
    "In the data set, there are several different cases where the booster did not land successfully. Sometimes a landing was attempted but failed due to an accident; for example, <code>True Ocean</code> means the mission outcome was successfully  landed to a specific region of the ocean while <code>False Ocean</code> means the mission outcome was unsuccessfully landed to a specific region of the ocean. <code>True RTLS</code> means the mission outcome was successfully  landed to a ground pad <code>False RTLS</code> means the mission outcome was unsuccessfully landed to a ground pad.<code>True ASDS</code> means the mission outcome was successfully landed on  a drone ship <code>False ASDS</code> means the mission outcome was unsuccessfully landed on a drone ship. \n",
    "\n",
    "In this lab we will mainly convert those outcomes into Training Labels with `1` means the booster successfully landed `0` means it was unsuccessful.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Falcon 9 first stage will land successfully\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/Images/landing_1.gif)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Several examples of an unsuccessful landing are shown here:\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/Images/crash.gif)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "   \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Objectives\n",
    "Perform exploratory  Data Analysis and determine Training Labels \n",
    "\n",
    "- Exploratory Data Analysis\n",
    "- Determine Training Labels \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "----\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Import Libraries and Define Auxiliary Functions\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We will import the following libraries.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Pandas is a software library written for the Python programming language for data manipulation and analysis.\n",
    "import pandas as pd\n",
    "#NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Data Analysis \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Load Space X dataset, from last section.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>FlightNumber</th>\n",
       "      <th>Date</th>\n",
       "      <th>BoosterVersion</th>\n",
       "      <th>PayloadMass</th>\n",
       "      <th>Orbit</th>\n",
       "      <th>LaunchSite</th>\n",
       "      <th>Outcome</th>\n",
       "      <th>Flights</th>\n",
       "      <th>GridFins</th>\n",
       "      <th>Reused</th>\n",
       "      <th>Legs</th>\n",
       "      <th>LandingPad</th>\n",
       "      <th>Block</th>\n",
       "      <th>ReusedCount</th>\n",
       "      <th>Serial</th>\n",
       "      <th>Longitude</th>\n",
       "      <th>Latitude</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>2010-06-04</td>\n",
       "      <td>Falcon 9</td>\n",
       "      <td>6104.959412</td>\n",
       "      <td>LEO</td>\n",
       "      <td>CCAFS SLC 40</td>\n",
       "      <td>None None</td>\n",
       "      <td>1</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>B0003</td>\n",
       "      <td>-80.577366</td>\n",
       "      <td>28.561857</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>2012-05-22</td>\n",
       "      <td>Falcon 9</td>\n",
       "      <td>525.000000</td>\n",
       "      <td>LEO</td>\n",
       "      <td>CCAFS SLC 40</td>\n",
       "      <td>None None</td>\n",
       "      <td>1</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>B0005</td>\n",
       "      <td>-80.577366</td>\n",
       "      <td>28.561857</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>2013-03-01</td>\n",
       "      <td>Falcon 9</td>\n",
       "      <td>677.000000</td>\n",
       "      <td>ISS</td>\n",
       "      <td>CCAFS SLC 40</td>\n",
       "      <td>None None</td>\n",
       "      <td>1</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>B0007</td>\n",
       "      <td>-80.577366</td>\n",
       "      <td>28.561857</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>2013-09-29</td>\n",
       "      <td>Falcon 9</td>\n",
       "      <td>500.000000</td>\n",
       "      <td>PO</td>\n",
       "      <td>VAFB SLC 4E</td>\n",
       "      <td>False Ocean</td>\n",
       "      <td>1</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>B1003</td>\n",
       "      <td>-120.610829</td>\n",
       "      <td>34.632093</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>2013-12-03</td>\n",
       "      <td>Falcon 9</td>\n",
       "      <td>3170.000000</td>\n",
       "      <td>GTO</td>\n",
       "      <td>CCAFS SLC 40</td>\n",
       "      <td>None None</td>\n",
       "      <td>1</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>B1004</td>\n",
       "      <td>-80.577366</td>\n",
       "      <td>28.561857</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>6</td>\n",
       "      <td>2014-01-06</td>\n",
       "      <td>Falcon 9</td>\n",
       "      <td>3325.000000</td>\n",
       "      <td>GTO</td>\n",
       "      <td>CCAFS SLC 40</td>\n",
       "      <td>None None</td>\n",
       "      <td>1</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>B1005</td>\n",
       "      <td>-80.577366</td>\n",
       "      <td>28.561857</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>7</td>\n",
       "      <td>2014-04-18</td>\n",
       "      <td>Falcon 9</td>\n",
       "      <td>2296.000000</td>\n",
       "      <td>ISS</td>\n",
       "      <td>CCAFS SLC 40</td>\n",
       "      <td>True Ocean</td>\n",
       "      <td>1</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>B1006</td>\n",
       "      <td>-80.577366</td>\n",
       "      <td>28.561857</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>8</td>\n",
       "      <td>2014-07-14</td>\n",
       "      <td>Falcon 9</td>\n",
       "      <td>1316.000000</td>\n",
       "      <td>LEO</td>\n",
       "      <td>CCAFS SLC 40</td>\n",
       "      <td>True Ocean</td>\n",
       "      <td>1</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>B1007</td>\n",
       "      <td>-80.577366</td>\n",
       "      <td>28.561857</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>9</td>\n",
       "      <td>2014-08-05</td>\n",
       "      <td>Falcon 9</td>\n",
       "      <td>4535.000000</td>\n",
       "      <td>GTO</td>\n",
       "      <td>CCAFS SLC 40</td>\n",
       "      <td>None None</td>\n",
       "      <td>1</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>B1008</td>\n",
       "      <td>-80.577366</td>\n",
       "      <td>28.561857</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>10</td>\n",
       "      <td>2014-09-07</td>\n",
       "      <td>Falcon 9</td>\n",
       "      <td>4428.000000</td>\n",
       "      <td>GTO</td>\n",
       "      <td>CCAFS SLC 40</td>\n",
       "      <td>None None</td>\n",
       "      <td>1</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>B1011</td>\n",
       "      <td>-80.577366</td>\n",
       "      <td>28.561857</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   FlightNumber        Date BoosterVersion  PayloadMass Orbit    LaunchSite  \\\n",
       "0             1  2010-06-04       Falcon 9  6104.959412   LEO  CCAFS SLC 40   \n",
       "1             2  2012-05-22       Falcon 9   525.000000   LEO  CCAFS SLC 40   \n",
       "2             3  2013-03-01       Falcon 9   677.000000   ISS  CCAFS SLC 40   \n",
       "3             4  2013-09-29       Falcon 9   500.000000    PO   VAFB SLC 4E   \n",
       "4             5  2013-12-03       Falcon 9  3170.000000   GTO  CCAFS SLC 40   \n",
       "5             6  2014-01-06       Falcon 9  3325.000000   GTO  CCAFS SLC 40   \n",
       "6             7  2014-04-18       Falcon 9  2296.000000   ISS  CCAFS SLC 40   \n",
       "7             8  2014-07-14       Falcon 9  1316.000000   LEO  CCAFS SLC 40   \n",
       "8             9  2014-08-05       Falcon 9  4535.000000   GTO  CCAFS SLC 40   \n",
       "9            10  2014-09-07       Falcon 9  4428.000000   GTO  CCAFS SLC 40   \n",
       "\n",
       "       Outcome  Flights  GridFins  Reused   Legs LandingPad  Block  \\\n",
       "0    None None        1     False   False  False        NaN    1.0   \n",
       "1    None None        1     False   False  False        NaN    1.0   \n",
       "2    None None        1     False   False  False        NaN    1.0   \n",
       "3  False Ocean        1     False   False  False        NaN    1.0   \n",
       "4    None None        1     False   False  False        NaN    1.0   \n",
       "5    None None        1     False   False  False        NaN    1.0   \n",
       "6   True Ocean        1     False   False   True        NaN    1.0   \n",
       "7   True Ocean        1     False   False   True        NaN    1.0   \n",
       "8    None None        1     False   False  False        NaN    1.0   \n",
       "9    None None        1     False   False  False        NaN    1.0   \n",
       "\n",
       "   ReusedCount Serial   Longitude   Latitude  \n",
       "0            0  B0003  -80.577366  28.561857  \n",
       "1            0  B0005  -80.577366  28.561857  \n",
       "2            0  B0007  -80.577366  28.561857  \n",
       "3            0  B1003 -120.610829  34.632093  \n",
       "4            0  B1004  -80.577366  28.561857  \n",
       "5            0  B1005  -80.577366  28.561857  \n",
       "6            0  B1006  -80.577366  28.561857  \n",
       "7            0  B1007  -80.577366  28.561857  \n",
       "8            0  B1008  -80.577366  28.561857  \n",
       "9            0  B1011  -80.577366  28.561857  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df=pd.read_csv(\"https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv\")\n",
    "df.head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Identify and calculate the percentage of the missing values in each attribute\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "FlightNumber       0.000000\n",
       "Date               0.000000\n",
       "BoosterVersion     0.000000\n",
       "PayloadMass        0.000000\n",
       "Orbit              0.000000\n",
       "LaunchSite         0.000000\n",
       "Outcome            0.000000\n",
       "Flights            0.000000\n",
       "GridFins           0.000000\n",
       "Reused             0.000000\n",
       "Legs               0.000000\n",
       "LandingPad        28.888889\n",
       "Block              0.000000\n",
       "ReusedCount        0.000000\n",
       "Serial             0.000000\n",
       "Longitude          0.000000\n",
       "Latitude           0.000000\n",
       "dtype: float64"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()/len(df)*100"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Identify which columns are numerical and categorical:\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "FlightNumber        int64\n",
       "Date               object\n",
       "BoosterVersion     object\n",
       "PayloadMass       float64\n",
       "Orbit              object\n",
       "LaunchSite         object\n",
       "Outcome            object\n",
       "Flights             int64\n",
       "GridFins             bool\n",
       "Reused               bool\n",
       "Legs                 bool\n",
       "LandingPad         object\n",
       "Block             float64\n",
       "ReusedCount         int64\n",
       "Serial             object\n",
       "Longitude         float64\n",
       "Latitude          float64\n",
       "dtype: object"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.dtypes"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### TASK 1: Calculate the number of launches on each site\n",
    "\n",
    "The data contains several Space X  launch facilities: <a href='https://en.wikipedia.org/wiki/List_of_Cape_Canaveral_and_Merritt_Island_launch_sites'>Cape Canaveral Space</a> Launch Complex 40  <b>VAFB SLC 4E </b> , Vandenberg Air Force Base Space Launch Complex 4E <b>(SLC-4E)</b>, Kennedy Space Center Launch Complex 39A <b>KSC LC 39A </b>.The location of each Launch Is placed in the column <code>LaunchSite</code>\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next, let's see the number of launches for each site.\n",
    "\n",
    "Use the method  <code>value_counts()</code> on the column <code>LaunchSite</code> to determine the number of launches  on each site: \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LaunchSite\n",
       "CCAFS SLC 40    55\n",
       "KSC LC 39A      22\n",
       "VAFB SLC 4E     13\n",
       "dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Apply value_counts() on column LaunchSite\n",
    "\n",
    "df.value_counts(df['LaunchSite'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Each launch aims to an dedicated orbit, and here are some common orbit types:\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\n",
    "\n",
    "* <b>LEO</b>: Low Earth orbit (LEO)is an Earth-centred orbit with an altitude of 2,000 km (1,200 mi) or less (approximately one-third of the radius of Earth),[1] or with at least 11.25 periods per day (an orbital period of 128 minutes or less) and an eccentricity less than 0.25.[2] Most of the manmade objects in outer space are in LEO <a href='https://en.wikipedia.org/wiki/Low_Earth_orbit'>[1]</a>.\n",
    "\n",
    "* <b>VLEO</b>: Very Low Earth Orbits (VLEO) can be defined as the orbits with a mean altitude below 450 km. Operating in these orbits can provide a number of benefits to Earth observation spacecraft as the spacecraft operates closer to the observation<a href='https://www.researchgate.net/publication/271499606_Very_Low_Earth_Orbit_mission_concepts_for_Earth_Observation_Benefits_and_challenges'>[2]</a>.\n",
    "\n",
    "\n",
    "* <b>GTO</b> A geosynchronous orbit is a high Earth orbit that allows satellites to match Earth's rotation. Located at 22,236 miles (35,786 kilometers) above Earth's equator, this position is a valuable spot for monitoring weather, communications and surveillance. Because the satellite orbits at the same speed that the Earth is turning, the satellite seems to stay in place over a single longitude, though it may drift north to south,” NASA wrote on its Earth Observatory website <a  href=\"https://www.space.com/29222-geosynchronous-orbit.html\" >[3] </a>.\n",
    "\n",
    "\n",
    "* <b>SSO (or SO)</b>: It is a Sun-synchronous orbit  also called a heliosynchronous orbit is a nearly polar orbit around a planet, in which the satellite passes over any given point of the planet's surface at the same local mean solar time <a href=\"https://en.wikipedia.org/wiki/Sun-synchronous_orbit\">[4] <a>.\n",
    "    \n",
    "    \n",
    "    \n",
    "* <b>ES-L1 </b>:At the Lagrange points the gravitational forces of the two large bodies cancel out in such a way that a small object placed in orbit there is in equilibrium relative to the center of mass of the large bodies. L1 is one such point between the sun and the earth <a href=\"https://en.wikipedia.org/wiki/Lagrange_point#L1_point\">[5]</a> .\n",
    "    \n",
    "    \n",
    "* <b>HEO</b> A highly elliptical orbit, is an elliptic orbit with high eccentricity, usually referring to one around Earth <a href=\"https://en.wikipedia.org/wiki/Highly_elliptical_orbit\">[6]</a>.\n",
    "\n",
    "\n",
    "* <b> ISS </b> A modular space station (habitable artificial satellite) in low Earth orbit. It is a multinational collaborative project between five participating space agencies: NASA (United States), Roscosmos (Russia), JAXA (Japan), ESA (Europe), and CSA (Canada)<a href=\"https://en.wikipedia.org/wiki/International_Space_Station\"> [7] </a>\n",
    "\n",
    "\n",
    "* <b> MEO </b> Geocentric orbits ranging in altitude from 2,000 km (1,200 mi) to just below geosynchronous orbit at 35,786 kilometers (22,236 mi). Also known as an intermediate circular orbit. These are \"most commonly at 20,200 kilometers (12,600 mi), or 20,650 kilometers (12,830 mi), with an orbital period of 12 hours <a href=\"https://en.wikipedia.org/wiki/List_of_orbits\"> [8] </a>\n",
    "\n",
    "\n",
    "* <b> HEO </b> Geocentric orbits above the altitude of geosynchronous orbit (35,786 km or 22,236 mi) <a href=\"https://en.wikipedia.org/wiki/List_of_orbits\"> [9] </a>\n",
    "\n",
    "\n",
    "* <b> GEO </b> It is a circular geosynchronous orbit 35,786 kilometres (22,236 miles) above Earth's equator and following the direction of Earth's rotation <a href=\"https://en.wikipedia.org/wiki/Geostationary_orbit\"> [10] </a>\n",
    "\n",
    "\n",
    "* <b> PO </b> It is one type of satellites in which a satellite passes above or nearly above both poles of the body being orbited (usually a planet such as the Earth <a href=\"https://en.wikipedia.org/wiki/Polar_orbit\"> [11] </a>\n",
    "\n",
    "some are shown in the following plot:\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/Images/Orbits.png)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### TASK 2: Calculate the number and occurrence of each orbit\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    " Use the method  <code>.value_counts()</code> to determine the number and occurrence of each orbit in the  column <code>Orbit</code>\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Orbit\n",
       "GTO      27\n",
       "ISS      21\n",
       "VLEO     14\n",
       "PO        9\n",
       "LEO       7\n",
       "SSO       5\n",
       "MEO       3\n",
       "ES-L1     1\n",
       "GEO       1\n",
       "HEO       1\n",
       "SO        1\n",
       "dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Apply value_counts on Orbit column\n",
    "\n",
    "df.value_counts(df['Orbit'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### TASK 3: Calculate the number and occurence of mission outcome of the orbits\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Use the method <code>.value_counts()</code> on the column <code>Outcome</code> to determine the number of <code>landing_outcomes</code>.Then assign it to a variable landing_outcomes.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Outcome\n",
      "True ASDS      41\n",
      "None None      19\n",
      "True RTLS      14\n",
      "False ASDS      6\n",
      "True Ocean      5\n",
      "False Ocean     2\n",
      "None ASDS       2\n",
      "False RTLS      1\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# landing_outcomes = values on Outcome column\n",
    "\n",
    "landing_outcomes = df.value_counts(df['Outcome'])\n",
    "print(landing_outcomes)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<code>True Ocean</code> means the mission outcome was successfully  landed to a specific region of the ocean while <code>False Ocean</code> means the mission outcome was unsuccessfully landed to a specific region of the ocean. <code>True RTLS</code> means the mission outcome was successfully  landed to a ground pad <code>False RTLS</code> means the mission outcome was unsuccessfully landed to a ground pad.<code>True ASDS</code> means the mission outcome was successfully  landed to a drone ship <code>False ASDS</code> means the mission outcome was unsuccessfully landed to a drone ship. <code>None ASDS</code> and <code>None None</code> these represent a failure to land.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0 True ASDS\n",
      "1 None None\n",
      "2 True RTLS\n",
      "3 False ASDS\n",
      "4 True Ocean\n",
      "5 False Ocean\n",
      "6 None ASDS\n",
      "7 False RTLS\n"
     ]
    }
   ],
   "source": [
    "for i,outcome in enumerate(landing_outcomes.keys()):\n",
    "    print(i,outcome)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We create a set of outcomes where the second stage did not land successfully:\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'False ASDS', 'False Ocean', 'False RTLS', 'None ASDS', 'None None'}"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bad_outcomes=set(landing_outcomes.keys()[[1,3,5,6,7]])\n",
    "bad_outcomes"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### TASK 4: Create a landing outcome label from Outcome column\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Using the <code>Outcome</code>,  create a list where the element is zero if the corresponding  row  in  <code>Outcome</code> is in the set <code>bad_outcome</code>; otherwise, it's one. Then assign it to the variable <code>landing_class</code>:\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "# landing_class = 0 if bad_outcome\n",
    "# landing_class = 1 otherwise\n",
    "\n",
    "landing_class = []\n",
    "for outcome in df['Outcome']:\n",
    "    if outcome in bad_outcomes:\n",
    "        landing_class.append(0)\n",
    "    else:\n",
    "        landing_class.append(1)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This variable will represent the classification variable that represents the outcome of each launch. If the value is zero, the  first stage did not land successfully; one means  the first stage landed Successfully \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Class\n",
       "0      0\n",
       "1      0\n",
       "2      0\n",
       "3      0\n",
       "4      0\n",
       "5      0\n",
       "6      1\n",
       "7      1"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Class']=landing_class\n",
    "df[['Class']].head(8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>FlightNumber</th>\n",
       "      <th>Date</th>\n",
       "      <th>BoosterVersion</th>\n",
       "      <th>PayloadMass</th>\n",
       "      <th>Orbit</th>\n",
       "      <th>LaunchSite</th>\n",
       "      <th>Outcome</th>\n",
       "      <th>Flights</th>\n",
       "      <th>GridFins</th>\n",
       "      <th>Reused</th>\n",
       "      <th>Legs</th>\n",
       "      <th>LandingPad</th>\n",
       "      <th>Block</th>\n",
       "      <th>ReusedCount</th>\n",
       "      <th>Serial</th>\n",
       "      <th>Longitude</th>\n",
       "      <th>Latitude</th>\n",
       "      <th>Class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>2010-06-04</td>\n",
       "      <td>Falcon 9</td>\n",
       "      <td>6104.959412</td>\n",
       "      <td>LEO</td>\n",
       "      <td>CCAFS SLC 40</td>\n",
       "      <td>None None</td>\n",
       "      <td>1</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>B0003</td>\n",
       "      <td>-80.577366</td>\n",
       "      <td>28.561857</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>2012-05-22</td>\n",
       "      <td>Falcon 9</td>\n",
       "      <td>525.000000</td>\n",
       "      <td>LEO</td>\n",
       "      <td>CCAFS SLC 40</td>\n",
       "      <td>None None</td>\n",
       "      <td>1</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>B0005</td>\n",
       "      <td>-80.577366</td>\n",
       "      <td>28.561857</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>2013-03-01</td>\n",
       "      <td>Falcon 9</td>\n",
       "      <td>677.000000</td>\n",
       "      <td>ISS</td>\n",
       "      <td>CCAFS SLC 40</td>\n",
       "      <td>None None</td>\n",
       "      <td>1</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>B0007</td>\n",
       "      <td>-80.577366</td>\n",
       "      <td>28.561857</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>2013-09-29</td>\n",
       "      <td>Falcon 9</td>\n",
       "      <td>500.000000</td>\n",
       "      <td>PO</td>\n",
       "      <td>VAFB SLC 4E</td>\n",
       "      <td>False Ocean</td>\n",
       "      <td>1</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>B1003</td>\n",
       "      <td>-120.610829</td>\n",
       "      <td>34.632093</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>2013-12-03</td>\n",
       "      <td>Falcon 9</td>\n",
       "      <td>3170.000000</td>\n",
       "      <td>GTO</td>\n",
       "      <td>CCAFS SLC 40</td>\n",
       "      <td>None None</td>\n",
       "      <td>1</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>B1004</td>\n",
       "      <td>-80.577366</td>\n",
       "      <td>28.561857</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   FlightNumber        Date BoosterVersion  PayloadMass Orbit    LaunchSite  \\\n",
       "0             1  2010-06-04       Falcon 9  6104.959412   LEO  CCAFS SLC 40   \n",
       "1             2  2012-05-22       Falcon 9   525.000000   LEO  CCAFS SLC 40   \n",
       "2             3  2013-03-01       Falcon 9   677.000000   ISS  CCAFS SLC 40   \n",
       "3             4  2013-09-29       Falcon 9   500.000000    PO   VAFB SLC 4E   \n",
       "4             5  2013-12-03       Falcon 9  3170.000000   GTO  CCAFS SLC 40   \n",
       "\n",
       "       Outcome  Flights  GridFins  Reused   Legs LandingPad  Block  \\\n",
       "0    None None        1     False   False  False        NaN    1.0   \n",
       "1    None None        1     False   False  False        NaN    1.0   \n",
       "2    None None        1     False   False  False        NaN    1.0   \n",
       "3  False Ocean        1     False   False  False        NaN    1.0   \n",
       "4    None None        1     False   False  False        NaN    1.0   \n",
       "\n",
       "   ReusedCount Serial   Longitude   Latitude  Class  \n",
       "0            0  B0003  -80.577366  28.561857      0  \n",
       "1            0  B0005  -80.577366  28.561857      0  \n",
       "2            0  B0007  -80.577366  28.561857      0  \n",
       "3            0  B1003 -120.610829  34.632093      0  \n",
       "4            0  B1004  -80.577366  28.561857      0  "
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can use the following line of code to determine  the success rate:\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"Class\"].mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can now export it to a CSV for the next section,but to make the answers consistent, in the next lab we will provide data in a pre-selected date range.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<code>df.to_csv(\"dataset_part_2.csv\", index=False)</code>\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Authors\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a href=\"https://www.linkedin.com/in/joseph-s-50398b136/\">Joseph Santarcangelo</a> has a PhD in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a href=\"https://www.linkedin.com/in/nayefaboutayoun/\">Nayef Abou Tayoun</a> is a Data Scientist at IBM and pursuing a Master of Management in Artificial intelligence degree at Queen's University.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Change Log\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "| Date (YYYY-MM-DD) | Version | Changed By | Change Description      |\n",
    "| ----------------- | ------- | ---------- | ----------------------- |\n",
    "| 2021-08-31        | 1.1     | Lakshmi Holla    | Changed Markdown |\n",
    "| 2020-09-20        | 1.0     | Joseph     | Modified Multiple Areas |\n",
    "| 2020-11-04        | 1.1.    | Nayef      | updating the input data |\n",
    "| 2021-05-026       | 1.1.    | Joseph      | updating the input data |\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Copyright © 2021 IBM Corporation. All rights reserved.\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python",
   "language": "python",
   "name": "conda-env-python-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
