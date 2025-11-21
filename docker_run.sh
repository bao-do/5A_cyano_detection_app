sudo docker rm label-studio
export LS_PORT=8080
export LOCAL_FILES_DOCUMENT_ROOT=/home/bao/School/5A/research_project/5A_cyano_detection_app/data
export LOCAL_SQL=$LOCAL_FILES_DOCUMENT_ROOT/ls_data
export LOCAL_SOURCE=$LOCAL_FILES_DOCUMENT_ROOT/ls_data/source_storage
export LOCAL_EXPORT=$LOCAL_FILES_DOCUMENT_ROOT/ls_data/target_storage
export LABEL_STUDIO_URL=http://0.0.0.0:$LS_PORT/
export LABEL_STUDIO_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA3MDk2NTE4MiwiaWF0IjoxNzYzNzY1MTgyLCJqdGkiOiIyMjc1ZDhjZjE5MGU0Y2M0YmMzYWJiN2VkYjRhMDEyMSIsInVzZXJfaWQiOjF9.pAMcDVKI7yCDvkYvP6mxJtoCN8GCeOAPGzd_i2fb-tc
sudo docker run -it --name label-studio \
  -p $LS_PORT:$LS_PORT \
  -v $LOCAL_SQL:/label-studio/data \
  -v $LOCAL_SOURCE:/label-studio/data/source_storage \
  -v $LOCAL_EXPORT:/label-studio/data/target_storage \
  -e LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
  -e LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/data \
  heartexlabs/label-studio:latest

  # sudo docker run -it -p 8080:8080 -v `pwd`/ls_data:/label-studio/data heartexlabs/label-studio:latest