"""Launch Google's installed Colab MCP with one IPv4 loopback socket.

The upstream server binds localhost with port=0. On Windows this creates
different ephemeral ports for ::1 and 127.0.0.1. The browser needs a single
advertised port. Keep origin checks, temporary tokens, and loopback isolation.
"""
import colab_mcp.session as session


class IPv4ColabWebSocketServer(session.ColabWebSocketServer):
    def __init__(self, host="127.0.0.1"):
        super().__init__(host="127.0.0.1")


session.ColabWebSocketServer = IPv4ColabWebSocketServer

if __name__ == "__main__":
    from colab_mcp import main
    main()
