"""GET com retry e backoff exponencial, compartilhado pelos clientes de API."""

import time
from collections.abc import Callable

import httpx

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


def get_with_retry(
    client: httpx.Client,
    url: str,
    *,
    retries: int = 3,
    backoff: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    """Faz GET; repete em erro de transporte ou status 429/5xx, até `retries` vezes extras.

    A espera dobra a cada tentativa (backoff, 2*backoff, ...). Erros 4xx não são repetidos.
    """
    for attempt in range(retries + 1):
        try:
            response = client.get(url)
            if response.status_code not in RETRY_STATUS:
                response.raise_for_status()
                return response
            error: Exception = httpx.HTTPStatusError(
                f"HTTP {response.status_code} em {url}", request=response.request, response=response
            )
        except httpx.TransportError as exc:
            error = exc
        if attempt == retries:
            raise error
        sleep(backoff * 2**attempt)
    raise AssertionError("inalcançável")
