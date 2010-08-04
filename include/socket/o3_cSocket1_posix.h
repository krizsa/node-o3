#ifndef SOCKET_V1_POSIX_H
#define SOCKET_V1_POSIX_H

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <fcntl.h>
#include <netdb.h>
#include <sys/select.h>

namespace o3 {
    struct cSocket1 : cSocket1Base {
        cSocket1() {

        }

        cSocket1(iCtx* ctx, int sock = -1, Type type = TYPE_INVALID, int state = 0) : m_sock(sock) 
		{
            // Make socket non-blocking
            int flags = fcntl(sock, F_GETFL, 0);
            
            flags |= O_NONBLOCK;
            fcntl(sock, F_SETFL, flags);
            m_type = type;
            m_state = state;
			m_ctx = ctx;
        }

        ~cSocket1()
        {
            if (m_timer_listener)
                m_timer_listener = 0;
            if (m_file_listener)
                m_file_listener = 0;
			if (m_state != STATE_CLOSED)
				::close(m_sock);	
        }

        o3_begin_class(cSocket1Base)
        o3_end_class();

		o3_glue_gen();

		static o3_ext("cO3") o3_fun siSocket socketUDP(iCtx* ctx)
		{			
			return create(ctx, TYPE_UDP);
		}

		static o3_ext("cO3") o3_fun siSocket socketTCP(iCtx* ctx) 
		{
			return create(ctx, TYPE_TCP);
		}

        
        static siSocket create(iCtx* ctx, Type type) 
		{		
            int sock;            

            switch (type) {
            case TYPE_TCP:    
                sock = ::socket(PF_INET, SOCK_STREAM, 0);
                break;
            case TYPE_UDP:
                sock = ::socket(PF_INET, SOCK_DGRAM, 0);
                break;
            };
            if (sock < 0)
                return siSocket();
            
			siScr scr = o3_new(cSocket1)(ctx, sock, type);
            return scr;
        }
        
        virtual bool bind(const char* host, int port) {
            m_addr.sin_family = AF_INET;
            hostent *hp = gethostbyname(host);
            if (!hp)
                return false;
            memCopy(&m_addr.sin_addr.s_addr, hp->h_addr, hp->h_length);
            m_addr.sin_port = htons(port);

            ::bind(m_sock, (sockaddr*) &m_addr, sizeof(struct sockaddr_in));
            return true;
        }
        
        virtual bool connect(const char* host, int port) {
            /*
             * Cannot connect if an error was raised or there still is a pending
             * connect or accept request.
             */
            if (m_state & (STATE_ERROR | STATE_CONNECTING | STATE_ACCEPTING))
                return false;
            
            /*
             * If the socket is already connected, close the previous connection
             * first. This will cause the all flags to be cleared.
             */
            if (m_state & STATE_CONNECTED)
                close();

            // Set up the address to connect to
            m_addr.sin_family = AF_INET;
            hostent *hp = gethostbyname(host);
            if (!hp)
                return false;
            memCopy(&m_addr.sin_addr.s_addr, hp->h_addr, hp->h_length);
            m_addr.sin_port = htons(port);

            m_bytes_sent        = 0;
            m_bytes_received    = 0;

            /*
             * Set the connecting flag. This will cause connect() to be called
             * in the callback function.
             */
            m_state = STATE_CONNECTING;

            // Set up a timer listener so that the callback will be called
			m_timer_listener = siCtx(m_ctx)->loop()->createTimer(100,
				Delegate(this, &cSocket1::trigger));								
        }
        
        virtual bool accept() {
            /*
             * Cannot accept if an error was raised or there still is a pending
             * connect or accept request.
             */
            if (m_state & (STATE_ERROR | STATE_CONNECTING | STATE_ACCEPTING))
                return false;

            /*
             * If the socket is already connected, close the previous connection
             * first. This will cause the all flags to be cleared.
             */
            if (m_state & STATE_CONNECTED)
                close();

            // Set up the address to accept from
            m_addr.sin_family = AF_INET;
            m_addr.sin_addr.s_addr = INADDR_ANY;
            m_addr.sin_port = 0; // INPORT_ANY

            ::listen(m_sock, 10);

            /*
             * Set the accepting flag. This will cause accept() to be called
             * in the callback function.
             */
            m_state = STATE_ACCEPTING;

            // Set up a timer listener so that the callback will be called
			m_timer_listener = siCtx(m_ctx)->loop()->createTimer(100,
				Delegate(this, &cSocket1::trigger));
        }
        
        virtual bool send(uint8_t* data, size_t size) {
            int err;

            /* 
             * Cannot send if an error was raised or the socket is not
             * connected.
             */
            if (m_state & STATE_ERROR || !(m_state & STATE_CONNECTED))
                return false;

            switch (m_type) {
            // For TCP sockets, we send the data as a stream
            case TYPE_TCP:
                // Append the data to be sent to the output buf
                m_out_buf.append(data, size);

                /*
                 * Set the sending flag. This will cause send() to be called
                 * in the callback function.
                 */
                m_state |= STATE_SENDING;
            
                // Set up a file listener so that the callback will be called
				if (!m_file_listener)
					m_file_listener = siCtx(m_ctx)->loop()->createListener(
						(void*) &m_sock, O_RDWR ,Delegate(this, &cSocket1::trigger));
				//m_timer_listener = siCtx(m_ctx)->loop()->createTimer(1,
				//	Delegate(this, &cSocket1::trigger));
				
				/*	
					scr_root->listenerLoop()->listen(
                   com_new(cHandle)((void*) m_sock), (iObj*) this,
                    (iListenerLoop::onTrigger) &cSocket1::trigger, 0); 
				*/
                break;
            // For UDP sockets, we send the data as a single packet
            case TYPE_UDP:
                err = ::send(m_sock, data, min(size, m_packet_size), 0);
                if (err < 0)
                    return false;
            };
            return true;
        }
        
        virtual bool sendTo(uint8_t* data, size_t size, const char* host, int port)
        {
            /*
             * Cannot send to if this is not a UDP socket or an error was
             * raised.
             */
            if (m_type != TYPE_UDP || (m_state & STATE_ERROR))
                return false;

            // Set up the address to send to
            m_addr.sin_family = AF_INET;
            hostent *hp = gethostbyname(host);
            if (!hp)
                return false;
            memCopy(&m_addr.sin_addr.s_addr, hp->h_addr, hp->h_length);
            m_addr.sin_port = htons(port);

            // For UDP sockets, we send the data as a single packet

            int err = ::sendto(m_sock, data, min(size, m_packet_size), 0,
                               (sockaddr*) &m_addr, sizeof(sockaddr_in));
            if (err < 0)          
                return false;
            return true;
        }
        
        virtual bool receive() {
            /* 
             * Cannot receive if an error was raised or the socket is not
             * connected.
             */
            if (m_state & STATE_ERROR || m_type == TYPE_TCP && !(m_state & STATE_CONNECTED))
                return false;

            /*
             * Set the receiving flag. This will cause recv() to be called
             * in the callback function.
             */
            m_state |= STATE_RECEIVING;
            // Set up a file listener so that the callback will be called
			if (!m_file_listener)
				m_file_listener = siCtx(m_ctx)->loop()->createListener(
					(void*)&m_sock, O_RDWR ,Delegate(this, &cSocket1::trigger));
            return true;
        }
        
        virtual void close() {
			if (m_timer_listener)
                m_timer_listener = 0;
            if (m_file_listener)
                m_file_listener = 0;	
			if (m_state != STATE_CLOSED)
				::close(m_sock);
            m_state = STATE_CLOSED;
        }

        void trigger(iUnk*) {
            if (m_state & STATE_ERROR || m_state & STATE_CLOSED) {
				m_timer_listener = 0;
                m_file_listener = 0;
            } else if (m_state & STATE_CONNECTING) {
                /*
                 * If the connecting flag is set, we try to call connect() on
                 * each callback.
                 */
                int err = ::connect(m_sock, (sockaddr*) &m_addr,
                                    sizeof(sockaddr_in));

                if (err < 0) {
                    switch (errno) {
                    case EALREADY:
                        /*
                         * If a previous call to connect was already done, we
                         * just retry on the next callback.
                         */
                        break;
                    case EISCONN:
                        /*
                         * If the socket is already connected, we assume that
                         * a previous call to connect succeeded just before the
                         * current one.
                         */
                        m_state = STATE_CONNECTED;
						Delegate(siCtx(m_ctx), m_on_connect)(this);
						return;
						break;
                    case EINPROGRESS:
                        /*
                         * If the current call to connect is still in progress,
                         * we just retry on the next callback.
                         */
                        break;
                    default:// In all other cases, we set the error flag
                        m_state |= STATE_ERROR;
                    }
					m_timer_listener->restart(100);
                } else {
                    /*
                     * If the call to connect() succeeds, the connecting flag is
                     * cleared, the connected flag is set, and the onConnect
                     * event is triggered.
                     */
                    m_state = STATE_CONNECTED;
					Delegate(siCtx(m_ctx), m_on_connect)(
						siScr(this));
                    m_timer_listener = 0; // We don't need anymore callbacks for now
                }
            } else if (m_state & STATE_ACCEPTING) {
                siScr on_accept;
                /*
                 * If the connecting flag is set, we try to call connect() on
                 * each callback.
                 */
                socklen_t addr_len = sizeof(sockaddr_in);
                int sock = ::accept(m_sock, (sockaddr*) &m_addr,
                                      &addr_len);
                
                if (sock < 0) {
                    switch (errno) {
                    case EWOULDBLOCK:
                        /*
                         * If the call to accept() would block because no
                         * incoming connections are available, we just retry on
                         * the next callback.
                         */
                        break;
                    default:
                        // In all other cases, we set the error flag
                        m_state |= STATE_ERROR;
                    }
                } else {
                    siScr scr = o3_new(cSocket1)(siCtx(m_ctx), sock, m_type, STATE_CONNECTED);
                  
                    Var arg(scr, g_sys);
					Delegate(siCtx(m_ctx), m_on_accept)(
						siScr(this));
                    m_timer_listener = 0; // We don't need anymore callbacks for now
                }
            } else if (m_state & (STATE_SENDING | STATE_RECEIVING)) {
            
                /*
                 * If either the sending or receiving flag is set, we need to
                 * figure out whether the socket is ready for reading or writing
                 * (or both), by calling select().
                 */
                fd_set  readfds;
                fd_set  writefds;
                fd_set  errorfds;
                timeval timeout;
                int     err;

                FD_ZERO(&readfds);
                FD_SET(m_sock, &readfds);
                FD_ZERO(&writefds);
                FD_SET(m_sock, &writefds);
                FD_ZERO(&errorfds);
                FD_SET(m_sock, &errorfds);
                timeout.tv_sec = 0;
                timeout.tv_usec = 0;
                do
                    err = ::select(m_sock + 1, &readfds, &writefds, &errorfds,
                                 &timeout);
                while (err < 0);

                if (m_state & STATE_SENDING && FD_ISSET(m_sock, &writefds)) {
                    /*
                     * If the sending flag is set and the socket is ready for
                     * writing, we send at most m_packet_size bytes from the
                     * output buf to the socket.
                     */
                    void*   data = m_out_buf.ptr();
                    size_t size = min(m_out_buf.size(), m_packet_size);

                    size = ::send(m_sock, data, size, 0);
                    if (size < 0) {
                        m_state |= STATE_ERROR;
                    } else {

                        /*
                         * If the call to send() succeeds, the onSend event is
                         * triggered, end the data sent is removed from the
                         * output buf. If the output buf becomes empty as a
                         * result, the sending bit is cleared as well.
                         */
                        m_bytes_sent += size;
						Delegate(siCtx(m_ctx), m_on_send)(
							siScr(this));
                        m_out_buf.remove(0, size);
                        if (m_out_buf.empty()) {
                            m_state &= ~STATE_SENDING;
                            if (!(m_state & STATE_RECEIVING))
                                m_file_listener = 0; // We don't need any more callbacks for now
                            }
                    }
                }
                if (m_state & STATE_RECEIVING && FD_ISSET(m_sock, &readfds)) {
					siScr on_receive;
                    /*
                     * If the receiving flag is set and the socket is ready for
                     * reading, we receive at most m_packet_size bytes from the
                     * socket to the input buf.
                     */
                    Buf buf;
                    
                    buf.reserve(m_packet_size);
                    sockaddr_in addr;
                    socklen_t addrlen = sizeof(struct sockaddr_in);
                    ssize_t size = ::recvfrom(m_sock, buf.ptr(), buf.capacity(), 0,
                                              (sockaddr*) &addr, &addrlen);
                    m_src_address = Str(inet_ntoa(addr.sin_addr));
                    if (size < 0) {
                        switch (errno) {
                        case EINTR:
                            /*
                             * If the call to recv() was interrupted, the socket
                             * will still be ready for reading on the next pass,
                             * so we will just retry on the next callback.
                             */
                            break;
                        default:
                            // In all other cases, we set the error flag
                            m_state |= STATE_ERROR;
                        }
                    } else if (size == 0) {
                        /*
                         * We assume that if the socket was ready for reading,
                         * but we received 0 bytes, that one side of the
                         * connection was closed. The most obvious way to deal
                         * with this seems to be to put the other end of the
                         * connection in an erroneous state.
                         */
                        m_state |= STATE_ERROR;
                    } else {
                        /*
                         * If the call to recv() succeeds, we append the
                         * received data to the end of the input buf, and
                         * trigger the onReceive event.
                         */
                        buf.resize(size);
                        m_received_buf.append(buf.ptr(), buf.size());
                        m_bytes_received += size;
						Delegate(siCtx(m_ctx), m_on_receive)(
							siScr(this));
                    }
                }
            }
        }

        // TODO: Create a file and timer listener for the socket
        int         m_sock;
        siTimer		m_timer_listener;
        siListener  m_file_listener;
        sockaddr_in m_addr;
        Buf         m_out_buf;
		siWeak		m_ctx;
    };
}

#endif // SOCKET_V1_POSIX_H
