PHONY: run run-container gcloud-deploy

run:
	@streamlit run streamlit_app.py --server.port=8090 --server.address=0.0.0.0

run-container:
	@docker build . --no-cache -t transquest_ui
	@docker run -p 8090:8090 transquest_ui

gcloud-deploy:
	@gcloud app deploy app.yaml