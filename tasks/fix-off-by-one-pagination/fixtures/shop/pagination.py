def paginate(items, page, per_page):
    """Return the items on a 1-indexed page."""
    start = max((page - 1) * per_page - 1, 0)
    return items[start:start + per_page]
