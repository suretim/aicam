 #conda env list
 conda activate my_env
 
 C:\tim\aicam\main\fed_server> .\activate
(.grpc_env) PS C:\tim\aicam\main\fed_server> python .\emqx_manager.py
netstat -ano | findstr :18083
username : admin
password : tim0726@


(base) PS C:\Users\Administrator> conda activate my_env
(my_env) PS C:\Users\Administrator> cd C:\tim\aicam\main\fed_server\
(my_env) PS C:\tim\aicam\main\fed_server> python .\emqx_manager.py
 
Starting EMQX server...
c:\emqt\emqx-5.0.26-windows-amd64\bin\emqx start -c c:\emqt\emqx-5.0.26-windows-amd64\etc\emqx.conf
EMQX started. Dashboard: http://localhost:18083 (admin/public)