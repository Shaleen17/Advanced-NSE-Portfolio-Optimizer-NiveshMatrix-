# Streamlit Cloud Deployment Steps

Use these steps after pushing the project to GitHub.

## Prerequisites

- GitHub repository contains `app.py` in the root folder.
- GitHub repository contains `requirements.txt` in the root folder.
- The `.streamlit/config.toml` file is committed for the black dashboard theme.
- The app runs locally with:

```bash
streamlit run app.py
```

## Deployment

1. Go to [Streamlit Community Cloud](https://share.streamlit.io).
2. Sign in with the GitHub account that owns the repository.
3. Click **Create app**.
4. Choose the GitHub repository.
5. Select the branch, usually `main`.
6. Set the app entrypoint file to:

```text
app.py
```

7. Open advanced settings only if you need to choose a specific Python version or add secrets.
8. Click **Deploy**.
9. Wait for dependencies to install from `requirements.txt`.
10. Open the generated `streamlit.app` URL and test the dashboard.

## Notes

- Do not move `app.py` out of the root unless you also update the Streamlit Cloud entrypoint.
- Keep `requirements.txt` small and accurate.
- If data download fails on cloud, test whether Yahoo Finance access is temporarily unavailable and retry later.
- This project is educational and is not financial advice.
