jupyter notebook --generate-config

python3
# >>> from notebook.auth import passwd; passwd();
# Enter password: cvspnb17
# Verify password: 
# 'sha1:1f9d642ea773:23880bdb1f7c7fb389d78a63f5e1233450a02493'


vi ~/.jupyter/jupyter_notebook_config.py
# c.NotebookApp.ip = '*'
# c.NotebookApp.password = u'sha1:bcd259ccf...<your hashed password here>'
# c.NotebookApp.open_browser = False