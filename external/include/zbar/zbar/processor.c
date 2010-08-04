/*------------------------------------------------------------------------
 *  Copyright 2007-2010 (c) Jeff Brown <spadix@users.sourceforge.net>
 *
 *  This file is part of the ZBar Bar Code Reader.
 *
 *  The ZBar Bar Code Reader is free software; you can redistribute it
 *  and/or modify it under the terms of the GNU Lesser Public License as
 *  published by the Free Software Foundation; either version 2.1 of
 *  the License, or (at your option) any later version.
 *
 *  The ZBar Bar Code Reader is distributed in the hope that it will be
 *  useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 *  of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser Public License
 *  along with the ZBar Bar Code Reader; if not, write to the Free
 *  Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 *  Boston, MA  02110-1301  USA
 *
 *  http://sourceforge.net/projects/zbar
 *------------------------------------------------------------------------*/

#include "processor.h"
//#include "window.h"
#include "image.h"

int _zbar_processor_unlock(zbar_processor_t *, int) { return 0;};
int _zbar_processor_lock(zbar_processor_t *) { return 0;};

static inline int proc_enter (zbar_processor_t *proc)
{
    _zbar_mutex_lock(&proc->mutex);
    return(_zbar_processor_lock(proc));
}

static inline int proc_leave (zbar_processor_t *proc)
{
    int rc = _zbar_processor_unlock(proc, 0);
    _zbar_mutex_unlock(&proc->mutex);
    return(rc);
}

static inline int proc_open (zbar_processor_t *proc)
{
    /* arbitrary default */
    int width = 640, height = 480;
    if(proc->video) {
        width = zbar_video_get_width(proc->video);
        height = zbar_video_get_height(proc->video);
    }
    return(_zbar_processor_open(proc, "zbar barcode reader", width, height));
}

/* API lock is already held */
int _zbar_process_image (zbar_processor_t *proc,
                         zbar_image_t *img)
{
    uint32_t force_fmt = proc->force_output;
    if(img) {
/*        if(proc->dumping) {
            zbar_image_write(proc->window->image, "zbar");
            proc->dumping = 0;
        }*/

        uint32_t format = zbar_image_get_format(img);
        zprintf(16, "processing: %.4s(%08" PRIx32 ") %dx%d @%p\n",
                (char*)&format, format,
                zbar_image_get_width(img), zbar_image_get_height(img),
                zbar_image_get_data(img));

        /* FIXME locking all other interfaces while processing is conservative
         * but easier for now and we don't expect this to take long...
         */
        zbar_image_t *tmp = zbar_image_convert(img, fourcc('Y','8','0','0'));
        if(!tmp)
            goto error;

        if(proc->syms) {
            zbar_symbol_set_ref(proc->syms, -1);
            proc->syms = NULL;
        }
        zbar_image_scanner_recycle_image(proc->scanner, img);
        int nsyms = zbar_scan_image(proc->scanner, tmp);
        _zbar_image_swap_symbols(img, tmp);

        zbar_image_destroy(tmp);
        tmp = NULL;
        if(nsyms < 0)
            goto error;

        proc->syms = zbar_image_scanner_get_results(proc->scanner);
        if(proc->syms)
            zbar_symbol_set_ref(proc->syms, 1);

/*        if(_zbar_verbosity >= 8) {
            const zbar_symbol_t *sym = zbar_image_first_symbol(img);
            while(sym) {
                zbar_symbol_type_t type = zbar_symbol_get_type(sym);
                int count = zbar_symbol_get_count(sym);
                zprintf(8, "%s%s: %s (%d pts) (dir=%d) (q=%d) (%s)\n",
                        zbar_get_symbol_name(type),
                        zbar_get_addon_name(type),
                        zbar_symbol_get_data(sym),
                        zbar_symbol_get_loc_size(sym),
                        zbar_symbol_get_orientation(sym),
                        zbar_symbol_get_quality(sym),
                        (count < 0) ? "uncertain" :
                        (count > 0) ? "duplicate" : "new");
                sym = zbar_symbol_next(sym);
            }
        }*/

        if(nsyms) {
            /* FIXME only call after filtering */
/*            _zbar_mutex_lock(&proc->mutex);

			_zbar_processor_notify(proc, EVENT_OUTPUT);  

			_zbar_mutex_unlock(&proc->mutex); */  // o3 -> stripped events
            if(proc->handler)
                proc->handler(img, proc->userdata);
        }

        if(force_fmt) {
            zbar_symbol_set_t *syms = img->syms;
            img = zbar_image_convert(img, force_fmt);
            if(!img)
                goto error;
            img->syms = syms;
            zbar_symbol_set_ref(syms, 1);
        }
    }

    /* display to window if enabled */
//    int rc = 0;
/*    if(proc->window) {
        if((rc = zbar_window_draw(proc->window, img)))
            err_copy(proc, proc->window);
        _zbar_processor_invalidate(proc);
    }*/

    if(force_fmt && img)
        zbar_image_destroy(img);
    return(0);

error:
    return(err_capture(proc, SEV_ERROR, ZBAR_ERR_UNSUPPORTED,
                       __func__, "unknown image format"));
}
/* o3 snip.. no GUI! 

int _zbar_processor_handle_input (zbar_processor_t *proc,
                                  int input)
{
    int event = EVENT_INPUT;
    switch(input) {
    case -1:
        event |= EVENT_CANCELED;
        _zbar_processor_set_visible(proc, 0);
        err_capture(proc, SEV_WARNING, ZBAR_ERR_CLOSED, __func__,
                    "user closed display window");
        break;

    case 'd':
        proc->dumping = 1;
        return(0);

    case '+':
    case '=':
        if(proc->window) {
            int ovl = zbar_window_get_overlay(proc->window);
            zbar_window_set_overlay(proc->window, ovl + 1);
        }
        break;

    case '-':
        if(proc->window) {
            int ovl = zbar_window_get_overlay(proc->window);
            zbar_window_set_overlay(proc->window, ovl - 1);
        }
        break;
    }

    _zbar_mutex_lock(&proc->mutex);
    proc->input = input;
    if(input == -1 && proc->visible && proc->streaming)
        // also cancel outstanding output waiters 
        event |= EVENT_OUTPUT;
    _zbar_processor_notify(proc, event);
    _zbar_mutex_unlock(&proc->mutex);
    return(input);
}
*/
#ifdef ZWANTTHREAD
static ZTHREAD proc_video_thread (void *arg)
{
    zbar_processor_t *proc = (zbar_processor_t *)arg;
    zbar_thread_t *thread = &proc->video_thread;

    _zbar_mutex_lock(&proc->mutex);
    _zbar_thread_init(thread);
    zprintf(4, "spawned video thread\n");

    while(thread->started) {
        /* wait for video stream to be active */
        while(thread->started && !proc->streaming)
            _zbar_event_wait(&thread->notify, &proc->mutex, NULL);
        if(!thread->started)
            break;

        /* blocking capture image from video */
        _zbar_mutex_unlock(&proc->mutex);
        zbar_image_t *img = zbar_video_next_image(proc->video);
        _zbar_mutex_lock(&proc->mutex);

        if(!img && !proc->streaming)
            continue;
        else if(!img)
            /* FIXME could abort streaming and keep running? */
            break;

        /* acquire API lock */
        _zbar_processor_lock(proc);
        _zbar_mutex_unlock(&proc->mutex);

        if(thread->started && proc->streaming)
            _zbar_process_image(proc, img);

        zbar_image_destroy(img);

        _zbar_mutex_lock(&proc->mutex);
        /* release API lock */
        _zbar_processor_unlock(proc, 0);
    }

    thread->running = 0;
    _zbar_event_trigger(&thread->activity);
    _zbar_mutex_unlock(&proc->mutex);
    return(0);
}

static ZTHREAD proc_input_thread (void *arg)
{
    zbar_processor_t *proc = (zbar_processor_t *)arg;
    zbar_thread_t *thread = &proc->input_thread;
    if(proc->window && proc_open(proc))
        goto done;

    _zbar_mutex_lock(&proc->mutex);
    thread->running = 1;
    _zbar_event_trigger(&thread->activity);
    zprintf(4, "spawned input thread\n");

    int rc = 0;
    while(thread->started && rc >= 0) {
        _zbar_mutex_unlock(&proc->mutex);
        rc = _zbar_processor_input_wait(proc, &thread->notify, -1);
        _zbar_mutex_lock(&proc->mutex);
    }

    _zbar_mutex_unlock(&proc->mutex);
    _zbar_processor_close(proc);
    _zbar_mutex_lock(&proc->mutex);

 done:
    thread->running = 0;
    _zbar_event_trigger(&thread->activity);
    _zbar_mutex_unlock(&proc->mutex);
    return(0);
}

#endif

zbar_processor_t *zbar_processor_create (int threaded)
{
    zbar_processor_t *proc = (zbar_processor_t *)calloc(1, sizeof(zbar_processor_t));
    if(!proc)
        return(NULL);
    err_init(&proc->err, ZBAR_MOD_PROCESSOR);

    proc->scanner = zbar_image_scanner_create();
    if(!proc->scanner) {
        free(proc);
        return(NULL);
    }

    proc->threaded = !_zbar_mutex_init(&proc->mutex) && threaded;
    //_zbar_processor_init(proc);
    return(proc);
}

void zbar_processor_destroy (zbar_processor_t *proc)
{
    zbar_processor_init(proc, NULL, 0);

    if(proc->scanner) {
        zbar_image_scanner_destroy(proc->scanner);
        proc->scanner = NULL;
    }

    _zbar_mutex_destroy(&proc->mutex);

	//    _zbar_processor_cleanup(proc);

    assert(!proc->wait_head);
    assert(!proc->wait_tail);
    assert(!proc->wait_next);

   // proc_waiter_t *w, *next;

	/* o3 - removed threads/events
    for(w = proc->free_waiter; w; w = next) {
        next = w->next;
        _zbar_event_destroy(&w->notify);
        free(w);
    }*/

    err_cleanup(&proc->err);
    free(proc);
}

int zbar_processor_init (zbar_processor_t *proc,
                         const char *dev,
                         int enable_display)
{
//    if(proc->video)
  //      zbar_processor_set_active(proc, 0);

/*    if(proc->window && !proc->input_thread.started)
        _zbar_processor_close(proc);*/ // windowing uit.

    _zbar_mutex_lock(&proc->mutex);

//    _zbar_thread_stop(&proc->input_thread, &proc->mutex);
 //  _zbar_thread_stop(&proc->video_thread, &proc->mutex);

    _zbar_processor_lock(proc);
    _zbar_mutex_unlock(&proc->mutex);


// done:
    _zbar_mutex_lock(&proc->mutex);
    proc_leave(proc);
    return(0);
}

zbar_image_data_handler_t*
zbar_processor_set_data_handler (zbar_processor_t *proc,
                                 zbar_image_data_handler_t *handler,
                                 const void *userdata)
{
    zbar_image_data_handler_t *result = NULL;
    proc_enter(proc);

    result = proc->handler;
    proc->handler = handler;
    proc->userdata = userdata;

    proc_leave(proc);
    return(result);
}

void zbar_processor_set_userdata (zbar_processor_t *proc,
                                  void *userdata)
{
    _zbar_mutex_lock(&proc->mutex);
    proc->userdata = userdata;
    _zbar_mutex_unlock(&proc->mutex);
}

void *zbar_processor_get_userdata (const zbar_processor_t *proc)
{
    zbar_processor_t *ncproc = (zbar_processor_t*)proc;
    _zbar_mutex_lock(&ncproc->mutex);
    void *userdata = (void*)ncproc->userdata;
    _zbar_mutex_unlock(&ncproc->mutex);
    return(userdata);
}

int zbar_processor_set_config (zbar_processor_t *proc,
                               zbar_symbol_type_t sym,
                               zbar_config_t cfg,
                               int val)
{
    proc_enter(proc);
    int rc = zbar_image_scanner_set_config(proc->scanner, sym, cfg, val);
    proc_leave(proc);
    return(rc);
}

int zbar_processor_request_size (zbar_processor_t *proc,
                                 unsigned width,
                                 unsigned height)
{
    proc_enter(proc);
    proc->req_width = width;
    proc->req_height = height;
    proc_leave(proc);
    return(0);
}

int zbar_processor_request_interface (zbar_processor_t *proc,
                                      int ver)
{
    proc_enter(proc);
    proc->req_intf = ver;
    proc_leave(proc);
    return(0);
}

int zbar_processor_request_iomode (zbar_processor_t *proc,
                                   int iomode)
{
    proc_enter(proc);
    proc->req_iomode = iomode;
    proc_leave(proc);
    return(0);
}

int zbar_processor_force_format (zbar_processor_t *proc,
                                 unsigned long input,
                                 unsigned long output)
{
    proc_enter(proc);
    proc->force_input = input;
    proc->force_output = output;
    proc_leave(proc);
    return(0);
}

int zbar_processor_is_visible (zbar_processor_t *proc)
{
    proc_enter(proc);
    int visible = proc->window && proc->visible;
    proc_leave(proc);
    return(visible);
}

int zbar_processor_set_visible (zbar_processor_t *proc,
                                int visible)
{
    proc_enter(proc);
    _zbar_mutex_unlock(&proc->mutex);

    int rc = 0;
/*    if(proc->window) {
        if(proc->video)
            rc = _zbar_processor_set_size(proc,
                                          zbar_video_get_width(proc->video),
                                          zbar_video_get_height(proc->video));
        if(!rc)
            rc = _zbar_processor_set_visible(proc, visible);

        if(!rc)
            proc->visible = (visible != 0);
    }
    else if(visible)
        rc = err_capture(proc, SEV_ERROR, ZBAR_ERR_INVALID, __func__,
                         "processor display window not initialized");

  */
	_zbar_mutex_lock(&proc->mutex);
    proc_leave(proc);
    return(rc);
}

const zbar_symbol_set_t*
zbar_processor_get_results (const zbar_processor_t *proc)
{
    zbar_processor_t *ncproc = (zbar_processor_t*)proc;
    proc_enter(ncproc);
    const zbar_symbol_set_t *syms = proc->syms;
    if(syms)
        zbar_symbol_set_ref(syms, 1);
    proc_leave(ncproc);
    return(syms);
}



int zbar_process_image (zbar_processor_t *proc,
                        zbar_image_t *img)
{
//    proc_enter(proc);
//    _zbar_mutex_unlock(&proc->mutex);

    int rc = 0;
//    if(img && proc->window)
  //      rc = _zbar_processor_set_size(proc,
    //                                  zbar_image_get_width(img),
      //                                zbar_image_get_height(img));
    if(!rc) 
	{
        zbar_image_scanner_enable_cache(proc->scanner, 0);
        rc = _zbar_process_image(proc, img);
        if(proc->streaming)
            zbar_image_scanner_enable_cache(proc->scanner, 1);
    }

//    _zbar_mutex_lock(&proc->mutex);
  //  proc_leave(proc);
    return(rc);
}
