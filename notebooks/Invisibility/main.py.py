# Databricks notebook source
# MAGIC %md ## Invisibility Cloak with GANs
# MAGIC 
# MAGIC Prerequisites:
# MAGIC * A GPU-enabled cluster on Databricks.
# MAGIC * PyTorch installed with GPU support.
# MAGIC 
# MAGIC Changes to make everything work on Databricks:
# MAGIC * import notebook calls are replaced by %run commands
# MAGIC * In base_optins return parser.parse_args() is replaced with return parser.parse_args(['23', '35']) as per some advice on stack overflow

# COMMAND ----------

import numpy as np
import sys

import mlflow
import mlflow.sklearn

#Clear Memory
import gc
gc.collect()

import os
cwd = os.getcwd()
print("Current working directory is " + cwd)

# COMMAND ----------

# MAGIC %md ## Import Important Stuff

# COMMAND ----------

dbutils.notebook.run("/Users/evan@datainsights.de/GAN_Invisibility_Cloak/train", 50)

# COMMAND ----------

