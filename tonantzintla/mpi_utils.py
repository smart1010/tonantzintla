import numpy as np

try:
    from mpi4py import MPI
    HAVE_MPI = True
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except Exception as exc:  # serial fallback only for a true one-process run
    import os
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        raise RuntimeError("mpi4py is required for multi-process Slurm execution") from exc
    HAVE_MPI = False
    class _Dummy:
        def Get_rank(self): return 0
        def Get_size(self): return 1
        def Barrier(self): pass
        def bcast(self, x, root=0): return x
        def gather(self, x, root=0): return [x]
        def reduce(self, x, op=None, root=0): return x
        def allreduce(self, x, op=None): return x
        def Send(self, *a, **k): pass
        def Recv(self, *a, **k): pass
        def Isend(self, *a, **k): return self
        def Irecv(self, *a, **k): return self
        def Waitall(self, reqs): pass
    COMM = _Dummy(); RANK = 0; SIZE = 1

def decompose_1d(Nx, size=None, rank=None):
    if size is None: size = SIZE
    if rank is None: rank = RANK
    counts = [Nx // size] * size
    for i in range(Nx % size): counts[i] += 1
    starts = [0]*size
    for i in range(1,size): starts[i] = starts[i-1] + counts[i-1]
    return starts[rank], starts[rank] + counts[rank], counts[rank], starts, counts

def exchange_ghosts_x(U, comm=None, rank=None, size=None):
    """Exchange one X ghost layer with contiguous buffers."""
    if comm is None: comm = COMM
    if rank is None: rank = RANK
    if size is None: size = SIZE
    if size == 1: return
    from mpi4py import MPI as _MPI
    left  = rank-1 if rank>0 else _MPI.PROC_NULL
    right = rank+1 if rank<size-1 else _MPI.PROC_NULL
    send_L = np.ascontiguousarray(U[:,1,:,:])
    send_R = np.ascontiguousarray(U[:,-2,:,:])
    recv_L = np.empty_like(send_L)
    recv_R = np.empty_like(send_R)
    reqs = [
        comm.Isend(send_L,  dest=left,  tag=11),
        comm.Irecv(recv_L, source=left,  tag=22),
        comm.Isend(send_R, dest=right, tag=22),
        comm.Irecv(recv_R, source=right, tag=11),
    ]
    _MPI.Request.Waitall(reqs)
    if left  != _MPI.PROC_NULL: U[:,  0, :, :] = recv_L
    if right != _MPI.PROC_NULL: U[:, -1, :, :] = recv_R


def exchange_right_face_state_x(A, face_axis, comm=None, rank=None, size=None, tag=31):
    """Supply each rank with the right state owned by its right neighbor."""
    if comm is None: comm = COMM
    if rank is None: rank = RANK
    if size is None: size = SIZE
    if size == 1: return A
    from mpi4py import MPI as _MPI
    left = rank - 1 if rank > 0 else _MPI.PROC_NULL
    right = rank + 1 if rank < size - 1 else _MPI.PROC_NULL
    left_face = [slice(None)] * A.ndim
    right_face = [slice(None)] * A.ndim
    left_face[face_axis] = 0
    right_face[face_axis] = -1
    send = np.ascontiguousarray(A[tuple(left_face)])
    recv = np.empty_like(send)
    reqs = [comm.Isend(send, dest=left, tag=tag), comm.Irecv(recv, source=right, tag=tag)]
    _MPI.Request.Waitall(reqs)
    if right != _MPI.PROC_NULL: A[tuple(right_face)] = recv
    return A


def synchronize_left_owned_flux_x(F, face_axis, left_index, right_index,
                                  comm=None, rank=None, size=None, tag=32):
    """Use one bitwise-identical flux on every shared X interface."""
    if comm is None: comm = COMM
    if rank is None: rank = RANK
    if size is None: size = SIZE
    if size == 1: return F
    from mpi4py import MPI as _MPI
    left = rank - 1 if rank > 0 else _MPI.PROC_NULL
    right = rank + 1 if rank < size - 1 else _MPI.PROC_NULL
    left_face = [slice(None)] * F.ndim
    right_face = [slice(None)] * F.ndim
    left_face[face_axis] = left_index
    right_face[face_axis] = right_index
    send = np.ascontiguousarray(F[tuple(right_face)])
    recv = np.empty_like(send)
    reqs = [comm.Isend(send, dest=right, tag=tag), comm.Irecv(recv, source=left, tag=tag)]
    _MPI.Request.Waitall(reqs)
    if left != _MPI.PROC_NULL: F[tuple(left_face)] = recv
    return F

def apply_bc_mpi(U, physical_left, physical_right):
    """Copy BC in Y/Z; X copy BC only on true physical ends."""
    # Y
    U[:, :, 0, :]  = U[:, :, 1, :]
    U[:, :, -1, :] = U[:, :, -2, :]
    # Z (if present)
    if U.ndim == 4 and U.shape[-1] > 1:
        U[:, :, :, 0]  = U[:, :, :, 1]
        U[:, :, :, -1] = U[:, :, :, -2]
    # X physical ends only
    if physical_left:
        U[:, 0, :, :] = U[:, 1, :, :]
    if physical_right:
        U[:, -1, :, :] = U[:, -2, :, :]
    return U

def gather_global_x(U_local, nx_local, Ny, Nz, comm=None, rank=None, size=None):
    """Gather interior slabs into a global array on rank 0; return None on others."""
    if comm is None: comm = COMM
    if rank is None: rank = RANK
    if size is None: size = SIZE
    import numpy as np
    interior = np.ascontiguousarray(U_local[:,1:1+nx_local,:,:])  # (5,nx,Ny,Nz)
    if size == 1:
        return interior
    if rank == 0:
        parts = [interior]
        for r in range(1,size):
            nxi = comm.recv(source=r, tag=90)  # receive nx
            tmp = np.empty((U_local.shape[0], nxi, Ny, Nz), dtype=U_local.dtype)
            comm.Recv(tmp, source=r, tag=91)
            parts.append(tmp)
        return np.concatenate(parts, axis=1)
    else:
        comm.send(nx_local, dest=0, tag=90)
        comm.Send(interior, dest=0, tag=91)
        return None


def gather_scalar_x(A_local, nx_local, Ny, Nz, comm=None, rank=None, size=None):
    """Gather scalar or multi-component arrays defined on cell centers with X halos.
    A_local shape: (nx_local+2, Ny, Nz) or (nx_local+2, Ny, Nz, C).
    Returns global array on rank 0 with shape (Nx, Ny, Nz[, C]); None on others.
    """
    if comm is None: comm = COMM
    if rank is None: rank = RANK
    if size is None: size = SIZE
    import numpy as np
    if size == 1:
        return A_local[1:1+nx_local]  # strip halos
    interior = A_local[1:1+nx_local]
    if rank == 0:
        parts = [interior]
        for r in range(1, size):
            nxi = comm.recv(source=r, tag=92)  # number of x cells from rank r
            # Allocate receiver buffer using this rank's interior trailing dims
            shape_tail = interior.shape[1:]
            tmp = np.empty((nxi,) + shape_tail, dtype=A_local.dtype)
            comm.Recv(tmp, source=r, tag=93)
            parts.append(tmp)
        return np.concatenate(parts, axis=0)
    else:
        comm.send(nx_local, dest=0, tag=92)
        comm.Send(interior, dest=0, tag=93)
        return None


def gather_bins_x(local, nx_local, Ny, Nz, nbins, comm=None, rank=None, size=None):
    """Gather interior dust bins (nx,Ny,Nz,nbins) from (nx_local+2,Ny,Nz,nbins) along X onto rank 0."""
    if comm is None: comm = COMM
    if rank is None: rank = RANK
    if size is None: size = SIZE
    import numpy as np
    interior = np.ascontiguousarray(local[1:1+nx_local, :, :, :])
    if size == 1:
        return interior
    if rank == 0:
        parts = [interior]
        for r in range(1,size):
            nxi = comm.recv(source=r, tag=290)
            tmp = np.empty((nxi, Ny, Nz, nbins), dtype=local.dtype)
            comm.Recv(tmp, source=r, tag=291)
            parts.append(tmp)
        return np.concatenate(parts, axis=0)
    else:
        comm.send(nx_local, dest=0, tag=290)
        comm.Send(interior, dest=0, tag=291)
        return None


def exchange_ghosts_x_scalar(q, comm=None, rank=None, size=None):
    """Exchange one X ghost layer for a scalar field.
    Works for shapes (nx+2, Ny, Nz) and (nx+2, Ny, Nz, NBINS).
    """
    if comm is None: comm = COMM
    if rank is None: rank = RANK
    if size is None: size = SIZE
    if size == 1:
        return
    try:
        from mpi4py import MPI as _MPI
    except Exception:
        return
    left  = rank-1 if rank>0 else _MPI.PROC_NULL
    right = rank+1 if rank<size-1 else _MPI.PROC_NULL
    send_L = np.ascontiguousarray(q[1,   ...])
    send_R = np.ascontiguousarray(q[-2,  ...])
    recv_L = np.empty_like(send_L)
    recv_R = np.empty_like(send_R)
    reqs = [
        comm.Isend(send_L,  dest=left,  tag=1111),
        comm.Irecv(recv_L, source=left,  tag=2222),
        comm.Isend(send_R,  dest=right, tag=2222),
        comm.Irecv(recv_R, source=right, tag=1111),
    ]
    from mpi4py import MPI as _MPI2
    _MPI2.Request.Waitall(reqs)
    if left  != _MPI.PROC_NULL:
        q[0,  ...] = recv_L
    if right != _MPI.PROC_NULL:
        q[-1, ...] = recv_R
