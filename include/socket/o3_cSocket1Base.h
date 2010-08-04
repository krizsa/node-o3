namespace o3 {    
    struct iSocket : iUnk {
        enum Type {
            TYPE_INVALID,
            TYPE_UDP,
            TYPE_TCP
        };

        enum State {        
            STATE_ERROR = 1,
            STATE_CLOSED = 2,
            STATE_CREATED = 4,
            STATE_CONNECTED = 8,
            STATE_CONNECTING = 16,
            STATE_ACCEPTING = 32,
            STATE_RECEIVING = 64,
            STATE_SENDING = 128
        };

        virtual bool bind(const char* addr, int port) = 0;
        virtual bool accept() = 0;
        virtual bool connect(const char* host, int port) = 0;
        virtual bool receive() = 0;
        virtual bool send(uint8_t* data, size_t size) = 0;
        virtual bool sendTo(uint8_t* data, size_t size, const char* url, int port) = 0;
        virtual void close() = 0;
    };

    o3_iid(iSocket, 0xe7ff0650, 0xee4f, 0x4bb2, 0xa0, 0x8, 0x8, 0x8f, 0x5c, 0xa4, 0xc, 0x45);

    struct cSocket1Base: cScr, iSocket {
        cSocket1Base() : m_packet_size(1024),
                       m_bytes_sent(0),
                       m_bytes_received(0),            
                       m_state(0),
                       m_type(TYPE_INVALID) 
        {
        }

        virtual ~cSocket1Base() 
        {
        }

        o3_begin_class(cScr)    
            o3_add_iface(iSocket)
        o3_end_class();

		o3_glue_gen();

        o3_enum( "State",        
            STATE_ERROR = 1,
            STATE_CLOSED = 2,
            STATE_CREATED = 4,
            STATE_CONNECTED = 8,
            STATE_CONNECTING = 16,
            STATE_ACCEPTING = 32,
            STATE_RECEIVING = 64,
            STATE_SENDING = 128
        );

		o3_enum( "Type",
			TYPE_INVALID,
			TYPE_UDP,
			TYPE_TCP
		);       

        o3_get bool error()
        {
            return (m_state & STATE_ERROR) > 0;
        }

        o3_get bool isClosed() 
        {
            return (m_state & STATE_CLOSED) > 0;
        }

        o3_get bool isCreated() 
        {
            return (m_state & STATE_CREATED) > 0;
        }

        o3_get bool isConnected() 
        {
            return (m_state & STATE_CONNECTED) > 0;
        }

        o3_get bool isConnecting() 
        {
            return (m_state & STATE_CONNECTING) > 0;
        }

        o3_get bool isAccepting() 
        {
            return (m_state & STATE_ACCEPTING) > 0;
        }

        o3_get bool isReceiveing() 
        {
            return (m_state & STATE_RECEIVING) > 0;
        }

        o3_get bool isSending()
		{
            return (m_state & STATE_SENDING) > 0;
        }
                        
        o3_fun bool bind(int port)
        {
            return bind("0.0.0.0", port);
        }

        o3_fun bool bind(const char* full)
        {
            Str host;
            int port;
            parseUrl(full, host, port);
            return bind(host, port);
        }

        o3_fun bool bind(const char* full, int port) = 0;

        o3_fun bool connect(const char* full) 
        {
            Str host;
            int port;
            parseUrl(full, host, port);
            return connect(host, port);
        }

        o3_fun bool connect(const char* host, int port) = 0;

		o3_fun bool accept() = 0;

		o3_fun bool receive() = 0;

        o3_fun bool send(const char* data) 
        {
            return send((uint8_t*)data, strLen(data)*sizeof(char));
        }

        o3_fun bool send(iBuf* ibuf) 
        {            
            if (!ibuf)
                return false;

            return send((uint8_t*)ibuf->unwrap().ptr(), 
				ibuf->unwrap().size());
        }

        virtual bool send(uint8_t* data, size_t size) = 0;

        o3_fun bool sendTo(iBuf* ibuf, const char* host, int port)
        {
			host; port;
			// TODO: handle host and port
            if (!ibuf)
                return false;

            return send((uint8_t*)ibuf->unwrap().ptr(), 
				ibuf->unwrap().size());
        }

        o3_fun bool sendTo(iBuf* ibuf, const char* full_addr)
        {
            Str host;
            int port;
            parseUrl(full_addr, host, port);
            return sendTo(ibuf, host, port);
        }

        virtual bool sendTo(uint8_t* data, size_t 
            size, const char* url, int port) = 0;
        

        o3_get Buf receivedBuf() 
        {
			return m_received_buf; 
        }
        
        o3_get Str receivedText() 
		{
			size_t next_zero;
			int8_t zero=0;
			Buf buf(m_received_buf);
			while( NOT_FOUND != (next_zero 
				= buf.find(&zero,1)))
			{
				buf.remove(next_zero,1);
			}

			return buf;
        }

		o3_fun void clearBuf()
		{
			m_received_buf.clear();
		}
		

        bool parseUrl(const char* url, Str& host, int& port) 
		{
            const char* p;

            for (p = url; *p; ++p)
                if (*p == ':')
                    break;
            host = Str(url, p-url);
            port = Str(++p).toInt32();
            return true;
        }

		o3_get siScr onaccept()
		{
			return m_on_accept;
		}
		o3_set siScr setOnaccept(iScr* cb)
		{
			return m_on_accept = cb;
		}

		o3_get siScr onconnect()
		{
			return m_on_connect;
		}
		o3_set siScr setOnconnect(iScr* cb)
		{
			return m_on_connect = cb;
		}

		o3_get siScr onreceive()
		{
			return m_on_receive;
		}
		o3_set siScr setOnreceive(iScr* cb)
		{
			return m_on_receive = cb;
		}

		o3_get siScr onsend()
		{
			return m_on_send;
		}
		o3_set siScr setOnsend(iScr* cb)
		{
			return m_on_send = cb;
		}

		o3_fun void close() = 0;

        o3_prop     size_t  m_packet_size;
        o3_get		size_t  m_bytes_sent;
        o3_get		size_t  m_bytes_received;                
        o3_get      Type    m_type;
        o3_get		Str     m_src_address;
		
		siScr m_on_accept;
		siScr m_on_connect;
		siScr m_on_receive;
		siScr m_on_send;

        Buf     m_received_buf;
        int     m_state;
		siWeak	m_ctx;

    };     
}