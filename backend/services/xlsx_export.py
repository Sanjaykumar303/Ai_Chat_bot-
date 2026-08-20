"""
Turns database query rows into a downloadable .xlsx - see
services/export_intent.py for how a chat message is recognized as this
kind of request, and routes/chat.py for where this is called from.

Takes rows already returned by db_query_service.answer_database_question()
(the exact same SQL generation -> sql_guard -> read-only Postgres
execution pipeline services/chat_service.py's own DATABASE_QUERY answer
already uses) - this module never queries the database itself, and never
invents a row that wasn't actually returned. An empty result set is
written out honestly as "no matching records", not silently skipped or
padded.
"""

import datetime
import io
from decimal import Decimal

from openpyxl import Workbook
from openpyxl.styles import Font

CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

_EXCEL_SHEET_TITLE_MAX_LEN = 31  # Excel's own hard limit on a sheet name


def build_xlsx(sheet_title, rows):
    """Returns the raw .xlsx bytes for `rows` (a list of dicts, the exact
    shape services/db_client.py's execute_readonly_query already returns)
    under one sheet named `sheet_title`."""

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = (sheet_title or "Export")[:_EXCEL_SHEET_TITLE_MAX_LEN]

    if not rows:
        sheet.append(["No matching records were found."])
    else:
        headers = list(rows[0].keys())
        sheet.append(headers)
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        for row in rows:
            sheet.append([_excel_safe(row.get(column)) for column in headers])

    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _excel_safe(value):
    """openpyxl writes str/int/float/bool/None/date/datetime natively,
    but psycopg2 hands back money columns as decimal.Decimal, which it
    doesn't - converted to float rather than left to raise or silently
    misrender. Anything else unrecognized falls back to str() rather than
    failing the whole export over one odd column type."""

    if isinstance(value, Decimal):
        return float(value)
    if value is None or isinstance(value, (str, int, float, bool, datetime.date, datetime.datetime)):
        return value
    return str(value)
