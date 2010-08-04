/*
 * Copyright (C) 2010 Ajax.org BV
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or (at your option) any later
 * version.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this library; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */
#ifndef O3_T_LIST_H
#define O3_T_LIST_H

namespace o3 {

template<typename T>
class tList {
    struct Node {
        Node* prev;
        Node* next;
        T x;

        Node(Node* prev, Node* next, const T& x) : prev(prev), next(next), x(x)
        {
        }
    };

    struct Impl {
        unsigned ref_count;
        size_t size;
        Node* head;
        Node* tail;

        Impl() : ref_count(1), size(0), head((Node*) /*malloc*/ memAlloc(sizeof(Node))),
                 tail(head)
        {
            head->prev = 0;
        }
    };

    Impl* m_impl;

    void makeUnique()
    {
        if (m_impl->ref_count > 1) {
            tList tmp;

            tmp.append(((const tList*) this)->begin(),
					   ((const tList*) this)->end());
            o3::swap(*this, tmp);
        }
    }

    Node* removeImpl(Node* head, Node* tail)
    {
        Node* prev = head->prev;

        while (head != tail) {
            Node* next = head->next;

            o3_delete(head);
            --m_impl->size;
            head = next;
        }
        if (prev)
            prev->next = tail;
        else
            m_impl->head = tail;
        tail->prev = prev;
        return tail;
    }

public:
    class ConstIter {
        typename tList::Node* m_node;

    public:
        ConstIter(typename tList::Node* node = 0) : m_node(node)
        {
            o3_trace1 trace;
        }

        bool operator==(const ConstIter& that) const
        {
            o3_trace1 trace;

            return m_node == that.m_node;
        }

        bool operator!=(const ConstIter& that) const
        {
            o3_trace1 trace;

            return !(*this == that);
        }

        bool valid() const
        {
            o3_trace1 trace;

            return m_node;
        }

        const T& operator*() const
        {
            o3_trace1 trace;

            return m_node->x;
        }

        const T* operator->() const
        {
            o3_trace1 trace;

            return &m_node->x;
        }

        ConstIter& operator++()
        {
            o3_trace1 trace;

            m_node = m_node->next;
            return *this;
        }

        ConstIter operator++(int)
        {
            o3_trace1 trace;
            ConstIter tmp = *this;

            ++*this;
            return tmp;
        }

        ConstIter& operator--()
        {
            o3_trace1 trace;

            m_node = m_node->prev;
            return *this;
        }

        ConstIter operator--(int)
        {
            o3_trace1 trace;
            ConstIter tmp = *this;

            --*this;
            return tmp;
        }
    };

    struct Iter {
        friend class tList;

        typename tList::Node* m_node;

    public:
        Iter(typename tList::Node* node = 0) : m_node(node)
        {
            o3_trace1 trace;
        }

        bool operator==(const Iter& that) const
        {
            o3_trace1 trace;

            return m_node == that.m_node;
        }

        bool operator!=(const Iter& that) const
        {
            o3_trace1 trace;

            return !(*this == that);
        }

        bool valid() const
        {
            o3_trace1 trace;

            return m_node;
        }

        T& operator*() const
        {
            o3_trace1 trace;

            return m_node->x;
        }

        T* operator->() const
        {
            o3_trace1 trace;

            return &m_node->x;
        }

        operator ConstIter() const
        {
            o3_trace1 trace;

            return m_node;
        }

        Iter& operator++()
        {
            o3_trace1 trace;

            m_node = m_node->next;
            return *this;
        }

        Iter operator++(int)
        {
            o3_trace1 trace;
            Iter tmp = *this;

            ++*this;
            return tmp;
        }

        Iter& operator--()
        {
            o3_trace1 trace;

            m_node = m_node->prev;
            return *this;
        }

        Iter operator--(int)
        {
            o3_trace1 trace;
            Iter tmp = *this;

            --*this;
            return tmp;
        }
    };

    tList() : m_impl(o3_new(Impl)())
    {
        o3_trace1 trace;
    }

    tList(const T& x, size_t n) : m_impl(o3_new(Impl)())
    {
        o3_trace1 trace;

        append(x, n);
    }

    tList(ConstIter first, ConstIter last) : m_impl(o3_new(Impl)())
    {
        o3_trace1 trace;

        append(first, last);
    }

    tList(const tList& that) : m_impl(that.m_impl)
    {
        o3_trace1 trace;

        ++m_impl->ref_count;
    }

    tList& operator=(const tList& that)
    {
        o3_trace1 trace;
		
		if (this != &that) {
			tList tmp(that);

			swap(*this, tmp);
		}
		return *this;
    }

    ~tList()
    {
        o3_trace1 trace;

        if (--m_impl->ref_count == 0) {
            clear();
            o3_delete(m_impl);
        }
    }

    bool empty() const
    {
        o3_trace1 trace;

        return size() == 0;
    }

    size_t size() const
    {
        o3_trace1 trace;

        return m_impl->size;
    }

    ConstIter begin() const
    {
        o3_trace1 trace;

        return m_impl->head;
    }

    ConstIter end() const
    {
        o3_trace1 trace;

        return m_impl->tail;
    }

    const T& front() const
    {
        o3_trace1 trace;

        return *begin();
    }

    const T& back() const
    {
        o3_trace1 trace;
        Iter tmp = end();

        return *--tmp;
    }

    Iter begin()
    {
        o3_trace1 trace;

        makeUnique();
        return m_impl->head;
    }

    Iter end()
    {
        o3_trace1 trace;

        makeUnique();
        return m_impl->tail;
    }

    T& front()
    {
        o3_trace1 trace;

        return *begin();
    }

    T& back()
    {
        o3_trace1 trace;
        Iter  tmp = end();

        return *--tmp;
    }

    Iter insert(Iter pos, const T& x)
    {
        o3_trace1 trace;
        Node* node = pos.m_node;
        Node* prev = node->prev;
        Node* new_node = o3_new(Node)(prev, node, x);

        ++m_impl->size;
        if (prev)
            prev->next = new_node;
        else {
            m_impl->head = new_node;
        }
        return node->prev = new_node;
    }

    void insert(Iter pos, const T& x, size_t n)
    {
        o3_trace1 trace;

        while (n--)
            pos = insert(pos, x);
    }

    void insert(Iter pos, ConstIter first, ConstIter last)
    {
        o3_trace1 trace;

        while (last != first)
            pos = insert(pos, *--last);
    }

    void append(const T& x, size_t n = 1)
    {
        o3_trace1 trace;

        insert(end(), x, n);
    }

    void append(ConstIter first, ConstIter last)
    {
        o3_trace1 trace;

        insert(end(), first, last);
    }

    Iter remove(Iter pos)
    {
        o3_trace1 trace;
        Iter tmp = pos;

        return removeImpl(pos.m_node, (++tmp).m_node);
    }

    void remove(Iter first, Iter last)
    {
        o3_trace1 trace;

        removeImpl(first.m_node, last.m_node);
    }

    void clear()
    {
        o3_trace1 trace;

        remove(begin(), end());
    }

    void pushFront(const T& x)
    {
        o3_trace1 trace;

        insert(begin(), x);
    }

    void pushBack(const T& x)
    {
        o3_trace1 trace;

        insert(end(), x);
    }

    void popFront()
    {
        o3_trace1 trace;

        remove(begin());
    }

    void popBack()
    {
        o3_trace1 trace;
        Iter tmp = end();

        remove(--tmp);
    }
};

}

#endif // O3_T_LIST_H
