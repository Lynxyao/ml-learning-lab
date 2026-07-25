from __future__ import annotations

import torch


def build_bipartite_laplacian(r_flat: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Build the ideal row-column network Laplacian from cell resistances.

    An n x n resistance map is represented by n row terminals and n column
    terminals. Cell (i, j) is the resistor connecting row i to column j.
    """
    if r_flat.ndim != 2:
        raise ValueError("r_flat must have shape [batch, n_cells]")

    batch_size, n_cells = r_flat.shape
    grid_size = int(round(n_cells**0.5))
    if grid_size * grid_size != n_cells:
        raise ValueError("number of cells must be a square")

    resistance = r_flat.reshape(batch_size, grid_size, grid_size)
    conductance = 1.0 / torch.clamp(resistance, min=eps)
    n_nodes = 2 * grid_size
    laplacian = torch.zeros(
        batch_size,
        n_nodes,
        n_nodes,
        dtype=r_flat.dtype,
        device=r_flat.device,
    )

    row_idx = torch.arange(grid_size, device=r_flat.device)
    col_idx = torch.arange(grid_size, device=r_flat.device)
    laplacian[:, row_idx, row_idx] = conductance.sum(dim=2)
    laplacian[:, grid_size + col_idx, grid_size + col_idx] = conductance.sum(dim=1)
    laplacian[:, :grid_size, grid_size:] = -conductance
    laplacian[:, grid_size:, :grid_size] = -conductance.transpose(1, 2)
    return laplacian


def ideal_pairwise_terminal_currents(
    r_flat: torch.Tensor,
    voltage: float = 5.0,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Compute n^2 ideal currents for row-column terminal-pair excitations.

    For each row i and column j, voltage is applied between those two
    terminals while all other terminals float. The returned current is the
    source current, so it represents an equivalent network response rather
    than the branch current through cell (i, j).

    This law-based implementation mirrors the assumptions used by Simon's
    current simulator. Those assumptions still require experimental validation.
    """
    laplacian = build_bipartite_laplacian(r_flat, eps=eps)
    batch_size, n_nodes, _ = laplacian.shape
    grid_size = n_nodes // 2
    source_voltage = torch.tensor(
        [voltage, 0.0],
        dtype=r_flat.dtype,
        device=r_flat.device,
    )
    all_idx = torch.arange(n_nodes, device=r_flat.device)
    currents = []

    for row in range(grid_size):
        for col in range(grid_size):
            driven = torch.tensor(
                [row, grid_size + col],
                dtype=torch.long,
                device=r_flat.device,
            )
            floating = all_idx[(all_idx != driven[0]) & (all_idx != driven[1])]

            l_dd = laplacian[:, driven][:, :, driven]
            l_df = laplacian[:, driven][:, :, floating]
            l_fd = laplacian[:, floating][:, :, driven]
            l_ff = laplacian[:, floating][:, :, floating]
            floating_response = torch.linalg.solve(l_ff, l_fd)
            reduced_laplacian = l_dd - torch.bmm(l_df, floating_response)
            driven_current = torch.matmul(reduced_laplacian, source_voltage)
            currents.append(driven_current[:, 0])

    return torch.stack(currents, dim=1).reshape(batch_size, grid_size * grid_size)


# Backward-compatible name used by earlier training scripts and checkpoints.
simon_forward_current_torch = ideal_pairwise_terminal_currents
