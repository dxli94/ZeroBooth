pip install diffusers==0.16

git checkout .
git pull

nohup streamlit run main.py > app.log 2>&1
