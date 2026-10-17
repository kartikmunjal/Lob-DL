import numpy as np

from lobdl.data.lobster import Message, OrderBook, SnapshotInitializedBook


def test_add_execute_delete_lifecycle():
    book = OrderBook()
    book.apply(Message(0.0, 1, 1, 10, 100, 1))
    book.apply(Message(0.1, 1, 2, 12, 102, -1))
    book.apply(Message(0.2, 4, 1, 4, 100, 1))
    assert book.bid[100] == 6
    book.apply(Message(0.3, 3, 2, 12, 102, -1))
    assert not book.ask


def test_snapshot_initialized_unattributed_rules():
    book = SnapshotInitializedBook(np.array([102, 12, 100, 10, 103, 8, 99, 7]))
    book.apply(Message(0.1, 2, 900, 4, 100, 1))
    assert book.bid[100] == 6
    book.apply(Message(0.2, 4, 901, 20, 102, -1))
    book.apply(Message(0.3, 3, 902, 2, 101, 1))
    book.apply(Message(0.4, 5, 903, 2, 102, -1))
    assert book.counts["unattributed_applied_events"] == 2
    assert book.counts["zero_clamps"] == 1
    assert book.counts["absent_level_skips"] == 1
    assert book.counts["hidden_executions"] == 1


def test_known_orders_are_attributed_after_snapshot():
    book = SnapshotInitializedBook(np.array([102, 12, 100, 10]))
    book.apply(Message(0.1, 1, 10, 5, 99, 1))
    book.apply(Message(0.2, 2, 10, 2, 99, 1))
    assert book.bid[99] == 3
    assert book.counts["attributed_events"] == 1


def test_crossed_state_is_logged_without_repair_or_reseed():
    book = SnapshotInitializedBook(np.array([102, 12, 100, 10]))
    book.apply(Message(0.1, 1, 10, 5, 103, 1))
    assert book.counts["crossed_or_locked_states"] == 1
    assert book.first_crossed_or_locked == {
        "message_index": 2,
        "seconds_since_midnight": 0.1,
    }
    assert book.bid[103] == 5
