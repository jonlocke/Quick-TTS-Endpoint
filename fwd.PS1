netsh interface portproxy delete v4tov4 listenaddress=127.0.0.1 listenport=8765
netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0   listenport=8765

netsh interface portproxy add v4tov4 listenaddress=127.0.0.1 listenport=8765 connectaddress=172.20.132.122 connectport=8765
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0   listenport=8765 connectaddress=172.20.132.122 connectport=8765

