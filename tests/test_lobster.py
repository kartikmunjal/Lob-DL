from lobdl.data.lobster import Message, OrderBook


def test_add_execute_delete_lifecycle():
    book = OrderBook()
    book.apply(Message(0.0, 1, 1, 10, 100, 1))
    book.apply(Message(0.1, 1, 2, 12, 102, -1))
    assert book.bid[100] == 10 and book.ask[102] == 12
    book.apply(Message(0.2, 4, 1, 4, 100, 1))
    assert book.bid[100] == 6
    book.apply(Message(0.3, 3, 2, 12, 102, -1))
    assert not book.ask
