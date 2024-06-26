{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "4b7a805e-dace-44b4-af6e-b19449ed73a2",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-05-24 07:09:30.086 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run /opt/anaconda3/lib/python3.11/site-packages/ipykernel_launcher.py [ARGUMENTS]\n",
      "2024-05-24 07:09:30.087 No runtime found, using MemoryCacheStorageManager\n",
      "2024-05-24 07:09:30.087 No runtime found, using MemoryCacheStorageManager\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "st.title('Uber pickups in NYC')\n",
    "\n",
    "DATE_COLUMN = 'date/time'\n",
    "DATA_URL = ('https://s3-us-west-2.amazonaws.com/'\n",
    "            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')\n",
    "\n",
    "@st.cache_data\n",
    "def load_data(nrows):\n",
    "    data = pd.read_csv(DATA_URL, nrows=nrows)\n",
    "    lowercase = lambda x: str(x).lower()\n",
    "    data.rename(lowercase, axis='columns', inplace=True)\n",
    "    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])\n",
    "    return data\n",
    "\n",
    "data_load_state = st.text('Loading data...')\n",
    "data = load_data(10000)\n",
    "data_load_state.text(\"Done! (using st.cache_data)\")\n",
    "\n",
    "if st.checkbox('Show raw data'):\n",
    "    st.subheader('Raw data')\n",
    "    st.write(data)\n",
    "\n",
    "st.subheader('Number of pickups by hour')\n",
    "hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]\n",
    "st.bar_chart(hist_values)\n",
    "\n",
    "# Some number in the range 0-23\n",
    "hour_to_filter = st.slider('hour', 0, 23, 17)\n",
    "filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]\n",
    "\n",
    "st.subheader('Map of all pickups at %s:00' % hour_to_filter)\n",
    "st.map(filtered_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7ef95f5e-0a81-4fe2-a79c-a7a7d5e4ac14",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
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
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
