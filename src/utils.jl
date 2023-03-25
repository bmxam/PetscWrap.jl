"""
Helps MPI print : execute "action" on each processor but one at a time. For instance
@one_at_a_time display(array).

Involves an MPI.Barrier
"""
macro one_at_a_time(action, comm=MPI.COMM_WORLD)
    return quote
        rank = MPI.Comm_rank($comm)
        nprocs = MPI.Comm_size($comm)
        for r in 0:nprocs-1
            if r == rank
                print("[$r] ")
                $(esc(action))
            end
            MPI.Barrier($comm)
        end
    end
end

"""
Execute `action` only on root processor, no MPI.Barrier
"""
macro only_root(action, comm=MPI.COMM_WORLD)
    return quote
        rank = MPI.Comm_rank($comm)
        if rank == 0
            print("[$rank] ")
            $(esc(action))
        end
    end
end

"""
Execute `action` only on `rank` processor, no MPI.Barrier
"""
macro only_proc(action, rank, comm=MPI.COMM_WORLD)
    return quote
        my_rank = MPI.Comm_rank($comm)
        if my_rank == $rank
            print("[$my_rank] ")
            $(esc(action))
        end
    end
end