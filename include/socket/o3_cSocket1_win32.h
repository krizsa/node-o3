//#include <windows.h>
//#include <winsock2.h>
#include <ws2tcpip.h>
#include <Mswsock.h>


namespace o3 {
    struct cSocket1: cSocket1Base {
        cSocket1() 
            : m_addr(0)
            , m_acc_addr(0)
            , m_to_addr(0)
            //, m_from_addr(0)
            , m_flags(0)                  
            , m_from_addr_length(sizeof(sockaddr_in))
        {
        }

        ~cSocket1()
		{
            if (m_addr)
                ::freeaddrinfo(m_addr);
            if (m_acc_addr)
                ::freeaddrinfo(m_acc_addr);
            if (m_to_addr)
                ::freeaddrinfo(m_to_addr);
            if (m_socket)
                ::closesocket(m_socket);
        }

        o3_begin_class(cSocket1Base)
            o3_add_iface(iSocket)
        o3_end_class();

		o3_glue_gen();

        int             m_stype;
        int             m_prot;
        siHandle        m_signal;
        siHandle        m_read_signal;
        siHandle        m_write_signal;
        siListener      m_listener;
        siListener      m_read_listener;
        siListener      m_write_listener;

        SOCKET           m_socket;
        struct addrinfo* m_addr;
        struct addrinfo* m_acc_addr;
        struct addrinfo* m_to_addr;        
        struct sockaddr_in m_from_addr;
        int              m_from_addr_length;
        WSAOVERLAPPED    m_overl_send;
        WSAOVERLAPPED    m_overl_receive;   
        OVERLAPPED       m_overl_conn;
        WSABUF           m_buf_in;
        WSABUF           m_buf_out;
        DWORD            m_flags;
        Buf             m_tmprecv_blob;

        tVec<Buf>    m_tosend;   
            
        //helper function for debugging, this should be done on cSys
        void startup() 
		{
            WSADATA wsd;
            WSAStartup(MAKEWORD(2,2), &wsd);
        }

		static o3_ext("cO3") o3_fun siSocket socketUDP(iCtx* ctx)
		{			
			return create(ctx, TYPE_UDP);
		}

		static o3_ext("cO3") o3_fun siSocket socketTCP(iCtx* ctx) 
		{
			return create(ctx, TYPE_TCP);
		}

        static siScr create(iCtx* ctx, Type type)
		{                                                            
            siScr ret = o3_new(cSocket1)();
            cSocket1* cret = (cSocket1*)ret.ptr();
            cret->m_type = type;
			cret->m_ctx = ctx;

            switch (type) {
                case TYPE_TCP:
                    cret->m_stype = SOCK_STREAM;
                    cret->m_prot = IPPROTO_TCP;
                    break;
                case TYPE_UDP:
                    cret->m_stype = SOCK_DGRAM;
                    cret->m_prot = IPPROTO_UDP;
                    break;
                default:
                    return siSocket();
            }                        
            
            cret->m_socket = WSASocket( AF_INET, cret->m_stype, cret->m_prot, 
                NULL, NULL, WSA_FLAG_OVERLAPPED);                        
            
            if (cret->m_socket == INVALID_SOCKET) {
                //int e = WSAGetLastError();
                return siSocket();
            }

            cret->setupEvent();
            cret->m_state = STATE_CREATED;
            return ret;
        }

        void setupEvent() 
		{
            HANDLE h;
            //create and set up a WSA event for the async connection
            h = WSACreateEvent();
            int e = WSAEventSelect(m_socket, h, FD_CONNECT|FD_ACCEPT|FD_CLOSE);  
            
            if(e) {                
                e = WSAGetLastError();
                o3_assert(false);
            }

			m_signal = o3_new(cHandle)(h, cHandle::TYPE_SOCKET);
			m_write_signal = o3_new(cHandle)(WSACreateEvent(), cHandle::TYPE_SOCKET);
            m_read_signal = o3_new(cHandle)(WSACreateEvent(), cHandle::TYPE_SOCKET);
            siCtx ctx(m_ctx);
			if (!ctx){
				m_state = STATE_ERROR;
				return;
			}

			m_listener = ctx->loop()->createListener(m_signal.ptr(), 1,
                         Delegate(this, &cSocket1::onevent));
            m_write_listener = ctx->loop()->createListener(m_write_signal.ptr(), 1,
                         Delegate(this, &cSocket1::onsend));
            m_read_listener = ctx->loop()->createListener(m_read_signal.ptr(), 1,
                         Delegate(this, &cSocket1::onreceive));
        }

        bool bind(const char* addr, int port) 
		{
            if (!m_socket)
                return false;
            
            //get address
			Str portstr = Str::fromInt32(port);
            
            if (m_addr)
                ::freeaddrinfo(m_addr);
            if (!(m_addr = getAddr(addr,portstr)))
                return false;

            //bind
            if (::bind(m_socket, m_addr->ai_addr,m_addr->ai_addrlen)) {
                //error
                //int e = WSAGetLastError();
                return false;
            }

            return true;
        }

        struct addrinfo* getAddr(const char* addr, const char* port) 
		{                
            struct addrinfo hints = {0}, *res;
            //struct sockaddr_in hints = {0};
            hints.ai_family = AF_INET;
            hints.ai_socktype = m_stype;
            hints.ai_protocol = m_prot;

            int rc = getaddrinfo(addr, port, &hints, &res);
            if (rc) {
                //sys_log(Str::format("getaddrinfo failed: %d\n", rc).ptr());
                //int e = WSAGetLastError();
                return 0;
            }
            return res;
        }

        void updateFromAddressString() 
		{            
            DWORD length = 512;
            m_src_address = Str();
            m_src_address.reserve(length);    
            int e = WSAAddressToStringA((LPSOCKADDR)&m_from_addr,m_from_addr_length,0,
                m_src_address.ptr(),  &length);
            if (e)
                m_src_address.resize(0);
            else
                m_src_address.resize(length);
        }

        bool connect(const char* url, int port)
		{            
            if (m_state & (STATE_ERROR | STATE_CONNECTING | STATE_ACCEPTING))
                return false;
                        
            if (m_state & STATE_CONNECTED)
                close();

            //get address
            struct addrinfo* to_addr;
			Str portstr = Str::fromInt32(port);
            m_src_address = Str(url); //+ :port, but it will be changed on the interface anyway

            if (!(to_addr = getAddr(url,portstr)))
                return false;
                                 
            //async connect:
            int err_connect, res_connect = WSAConnect(m_socket, 
                to_addr->ai_addr, to_addr->ai_addrlen,NULL, NULL, NULL, NULL);                       

            err_connect = WSAGetLastError();
            
            freeaddrinfo(to_addr);
            if ((res_connect != 0) && (err_connect == WSAEWOULDBLOCK)) {
                //async connection started
                m_state |= STATE_CONNECTING;
                return true;
            }

            if (!res_connect && !err_connect) {
                //immediate success
                m_state |= STATE_CONNECTING;
                onconnect();
                return true;
            }
    
            return false;
        }

        bool accept() 
		{
            if (m_state & (STATE_ERROR | STATE_CONNECTING | STATE_ACCEPTING))
                return false;

            if (m_state & STATE_CONNECTED)
                close();            

            int e  = listen(m_socket, 10);
            if(e) {
                e = WSAGetLastError();
                m_state |= STATE_ERROR;
            }else{
                m_state |= STATE_ACCEPTING;
                return true;
            }

            return false;
        }

        bool send(uint8_t* data, size_t size)
		{
            if (m_state & STATE_ERROR || !(m_state & STATE_CONNECTED))
                return false;
            
            m_tosend.push(Buf());
            m_tosend.back().append(data, size);
            
            if (m_state & STATE_SENDING)
                return true;
            
            DWORD bytes_sent = 0;
            int err_send, res_send = sendFirstBuf(size, bytes_sent);
            err_send = WSAGetLastError();

            if ((res_send == SOCKET_ERROR) && (err_send == WSAEWOULDBLOCK)) {
                //delayed, we can wait for the cb
                m_state |= STATE_SENDING;
                return true;
            }
            
            if (!res_send && bytes_sent) {
                //immediate success
                m_state |= STATE_SENDING;
                onsend(0);
                return true;
            }
                        
            return false;
        }

        virtual bool sendTo(uint8_t* data, size_t size, const char* url, int port) 
		{
            if (m_state & STATE_ERROR || m_type != TYPE_UDP)
                return false;

            m_tosend.push(Buf());
            m_tosend.back().append(data, size);

            //get address          
			Str portstr = Str::fromInt32(port);
            if (m_to_addr)
                ::freeaddrinfo(m_to_addr);
            if (!(m_to_addr = getAddr(url,portstr)))
                return false;
            
            // Create an event and an ovelapped struct for the async send            
            memSet(&m_overl_send, 0, sizeof(WSAOVERLAPPED));
            m_overl_send.hEvent = m_signal->handle();
                
            m_buf_out.len = size;
            m_buf_out.buf = (char*) m_tosend[0].ptr();

            //async send 
            DWORD bytes_sent = 0;
            int res_send = WSASendTo(m_socket, &m_buf_out, 1, &bytes_sent, 0, 
                m_to_addr->ai_addr, m_to_addr->ai_addrlen, &m_overl_send, NULL);    

            if (res_send == SOCKET_ERROR) {
                //send to must return immediately or we report an error
                m_state |= STATE_ERROR;
                return true;
            }
            
            if (!res_send && bytes_sent) {
                //immediate success
                m_state |= STATE_SENDING;
                onsend(0);
                return true;
            }

			return false;
        }

        int sendFirstBuf(size_t size, DWORD& bytes_sent)
		{
            // Create an event and an ovelapped struct for the async send            
            memSet(&m_overl_send, 0, sizeof(WSAOVERLAPPED));
            m_overl_send.hEvent = m_write_signal->handle();            
                
            //if its TCP and the data is too big lets send it chunk by chunk
            bool chunk = (m_type == TYPE_UDP) ? false : (size > m_packet_size);                        
            m_buf_out.len = chunk ? m_packet_size : size;
            m_buf_out.buf = (char*) m_tosend[0].ptr();

            //async send             
            return WSASend(m_socket, &m_buf_out, 1,
                     &bytes_sent, 0, &m_overl_send, NULL);
        }

        virtual bool receive() 
		{
            if (m_state & STATE_RECEIVING || m_state & STATE_ERROR 
                    || m_type == TYPE_TCP && !(m_state & STATE_CONNECTED))
                return false;
                            

            m_state |= STATE_RECEIVING;
            memSet(&m_overl_receive, 0, sizeof(WSAOVERLAPPED));
            //if (m_type == TYPE_UDP)
            //    return true;

            //async recv.
            int err, res = startnewrecv();
            err = WSAGetLastError();
            if ((res != 0) && (err == WSAEWOULDBLOCK || err == WSA_IO_PENDING)) {                
                m_state |= STATE_RECEIVING;
                return true;
            }

            if (!res && !err) {
                m_state |= STATE_RECEIVING;
                onreceive(0);
                return true;
            }
                        
            return false;            
        }

        int startnewrecv() 
		{
            m_tmprecv_blob.empty();
            m_tmprecv_blob.reserve(m_packet_size * 4);
            //m_received_blob.resize(0);            
            memSet(&m_overl_receive, 0, sizeof(WSAOVERLAPPED));
            
            m_overl_receive.hEvent = m_read_signal->handle();
                              
            m_buf_in.len = m_packet_size;
            m_buf_in.buf = (char*) m_tmprecv_blob.ptr();

            //async recv.
            //return WSARecv(m_socket, &m_buf_in, 1,
            //         NULL, &m_flags, &m_overl_receive, NULL);
            return WSARecvFrom(m_socket, &m_buf_in, 1, NULL, &m_flags, 
                (struct sockaddr*)&m_from_addr, &m_from_addr_length, &m_overl_receive, NULL);
        }

        void close() 
		{
            ::closesocket(m_socket);
            m_socket = 0;
            m_state = STATE_CLOSED;
            m_listener = 0;
        }

        void terminate() 
		{
        
        }

        void onevent(iUnk*) 
		{ 
            WSANETWORKEVENTS networkEvents;
            int e,res = WSAEnumNetworkEvents(m_socket,m_signal->handle(), &networkEvents);      
            if (res) {
                e = WSAGetLastError();            
            }

            long events = networkEvents.lNetworkEvents;
            if (events & FD_CONNECT)
                onconnect();                     
            if (events & FD_ACCEPT)
                onacc(0);
            if (events & FD_CLOSE)
                onclose();
            if (events & FD_OOB)
                onreceive(0);            //!CHECK:
            
            WSAResetEvent(m_signal->handle());
        }

        void onacc(iUnk*) 
		{             
            if (!(m_state & STATE_ACCEPTING))
                return;

            SOCKET acc = WSAAccept(m_socket, NULL /*m_acc_addr->ai_addr*/,0 /*(int*)&m_acc_addr->ai_addrlen*/
                ,NULL,NULL);
  
            if (acc == INVALID_SOCKET){
                //int e = WSAGetLastError();
                return;
            }

            //now we have a socket, lets create a cSocket around it and pass it in a Var
            //back to the callback function as an argument
            siSocket sacc = o3_new(cSocket1)();
            if (!sacc)
                return;

            cSocket1* csocc = (cSocket1*) sacc.ptr();
            csocc->m_ctx = siCtx(m_ctx);
			csocc->m_socket = acc;
            csocc->m_stype = m_stype;
            csocc->m_prot = m_prot;

            csocc->setupEvent();
            csocc->m_state = STATE_CREATED | STATE_CONNECTED;
                        
			Delegate(siCtx(m_ctx), m_on_accept)(
				siScr(csocc));
        }

        void onconnect()
		{ 
            if (! (m_state & STATE_CONNECTING))
                return;

            m_state &= ~STATE_CONNECTING;
            m_state |= STATE_CONNECTED;
			Delegate(siCtx(m_ctx), m_on_connect)(
				siScr(this));
        }

        void onreceive(iUnk*) 
		{ 
            WSAResetEvent(m_read_signal->handle());
            if (!(m_state & STATE_RECEIVING))
                return;

            //if ( m_overl_receive.hEvent == 0) {
            //    //lets start to receive    
            //    startnewrecv();
            //    onreceive();
            //    return;
            //}

            //check the result of the async op.
            DWORD bytes_received, flags(0);            
            BOOL res = WSAGetOverlappedResult(m_socket, &m_overl_receive, &bytes_received,
                                    FALSE, &flags);

            if (!res) {
                //int e = WSAGetLastError();
                m_state |= STATE_ERROR;
            }

            //set the new from address string
            if (m_type == TYPE_UDP)
                updateFromAddressString();

            //read the data
            m_bytes_received += bytes_received;            
            m_tmprecv_blob.resize(bytes_received);
            m_received_buf.append(m_tmprecv_blob.ptr(),
				m_tmprecv_blob.size()); 
                              
			Delegate(siCtx(m_ctx), m_on_receive)(
				siScr(this));


            memSet(&m_overl_receive, 0, sizeof(WSAOVERLAPPED));             
            res = startnewrecv();
            //int err = WSAGetLastError();
            //if (!res && !err) 
            //    onreceive(0);
        
        }

       void onsend(iUnk*) 
	   { 
            WSAResetEvent(m_write_signal->handle());
            if (!(m_state & STATE_SENDING))
                return;

            //lets check the result of the overlapped send operation:
            DWORD bytes_sent, flags(0);            
            BOOL res_op = WSAGetOverlappedResult(m_socket, &m_overl_send, &bytes_sent,
                                    FALSE, &flags);

            if (!res_op) {
                //int e = WSAGetLastError();
                m_state |= STATE_ERROR;
                return;
            }
            
            m_bytes_sent += bytes_sent;
            //!TODO: check the size of data
            Delegate(siCtx(m_ctx), m_on_send)(this);

            int err, res;

            //if we could not send the whole blob in one chunk lets send the next
            if (m_bytes_sent < m_tosend[0].size()) {
                if (m_type == TYPE_UDP) {
                    //in UDP mode there is no such thing as sending the message in chunks 
                    m_state = STATE_ERROR;
                     return;
                }

                //some of the first blob from the cue is already sent
                //lets send the next chunk               

                //overlapped struct for the overlapped send:
                memSet(&m_overl_send, 0, sizeof(WSAOVERLAPPED));
                m_overl_send.hEvent = m_write_signal->handle();
                //size and buffer pointer:
                size_t rest = m_tosend[0].size() - m_bytes_sent;
                bool chunk = rest > m_packet_size;                            
                m_buf_out.len = chunk ? m_packet_size : rest;
                m_buf_out.buf = (char*) m_tosend[0].ptr() + m_bytes_sent;
                //actual send:
                DWORD bytes_sent;
                res = WSASend(m_socket, &m_buf_out,
                    1,&bytes_sent, 0, &m_overl_send, NULL);
                //check the result:
                err = WSAGetLastError();
                if (res && (err != WSAEWOULDBLOCK || err != WSA_IO_PENDING))                                             
                    m_state = STATE_ERROR;
                if (!res && bytes_sent)
                    onsend(0);

            } else {
                //first blob from the cue is sent lets send the next if there is any
                m_tosend.remove(0);
                if (!m_tosend.empty()) {
                    //lets send the next blob
                    DWORD bytes_sent(0);
                    res = sendFirstBuf(m_tosend[0].size(),bytes_sent);
                    err = WSAGetLastError();
                    if (res && (err != WSAEWOULDBLOCK || err != WSA_IO_PENDING))                                             
                        m_state = STATE_ERROR;
                } else {
                    //no more blobs to send, lets finsih
                    m_state &= ~STATE_SENDING;
                }
            }
        }

        void onclose(){
        }

    };
}

