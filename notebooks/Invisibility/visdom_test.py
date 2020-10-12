# Databricks notebook source
import visdom
import numpy as np

# COMMAND ----------

# MAGIC %%system
# MAGIC python3 -m pip install visdom

# COMMAND ----------

# MAGIC %%bash
# MAGIC python3 -m visdom.server -port 8080 --hostname "10.35.247.75"

# COMMAND ----------

ssh ubuntu@ec2-18-194-176-66.eu-central-1.compute.amazonaws.com -p 2200 -i <private_key_file_path>

# COMMAND ----------

vis = visdom.Visdom("http://ec2-3-129-61-90.us-east-2.compute.amazonaws.com")
vis.text('Hey Marin, check this motthafukka out.')

# COMMAND ----------

