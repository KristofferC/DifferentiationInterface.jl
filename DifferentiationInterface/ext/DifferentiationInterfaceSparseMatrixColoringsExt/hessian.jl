struct SMCSparseHessianPrep{
        SIG,
        BS <: DI.BatchSizeSettings,
        P <: AbstractMatrix,
        C <: AbstractColoringResult{:symmetric, :column},
        M <: AbstractMatrix{<:Number},
        Sp <: NTuple,
        S <: AbstractVector{<:NTuple},
        R <: AbstractVector{<:NTuple},
        E2 <: DI.HVPPrep,
        DR <: Union{Nothing, NTuple},
    } <: DI.SparseHessianPrep{SIG}
    _sig::Val{SIG}
    batch_size_settings::BS
    sparsity::P
    coloring_result::C
    compressed_matrix::M
    batched_seed_prep::Sp
    batched_seeds::S
    batched_results::R
    hvp_prep::E2
    # throwaway HVP buffers, only present for an all-zero sparsity pattern:
    # no HVP batch ever runs there, so the gradient must be extracted with a
    # dedicated gradient_and_hvp! call (HVP-only backends have no gradient)
    degenerate_results::DR
end

## Hessian, one argument

function DI.prepare_hessian_nokwarg(
        strict::Val, f::F, backend::AutoSparse, x, contexts::Vararg{DI.Context, C}
    ) where {F, C}
    dense_backend = dense_ad(backend)
    sparsity = DI.hessian_sparsity_with_contexts(
        f, sparsity_detector(backend), x, contexts...
    )
    problem = ColoringProblem{:symmetric, :column}()
    coloring_result = coloring(
        sparsity, problem, coloring_algorithm(backend); decompression_eltype = eltype(x)
    )
    N = length(column_groups(coloring_result))
    batch_size_settings = DI.pick_batchsize(DI.outer(dense_backend), N)
    return _prepare_sparse_hessian_aux(
        strict, batch_size_settings, sparsity, coloring_result, f, backend, x, contexts...
    )
end

function _prepare_sparse_hessian_aux(
        strict::Val,
        batch_size_settings::DI.BatchSizeSettings{B},
        sparsity::AbstractMatrix,
        coloring_result::AbstractColoringResult{:symmetric, :column},
        f::F,
        backend::AutoSparse,
        x,
        contexts::Vararg{DI.Context, C}
    ) where {B, F, C}
    _sig = DI.signature(f, backend, x, contexts...; strict)
    (; N, A) = batch_size_settings
    dense_backend = dense_ad(backend)
    groups = column_groups(coloring_result)
    seed_prep = DI.multibasis(x, eachindex(x))
    seeds = [DI.multibasis(x, eachindex(x)[group]) for group in groups]
    compressed_matrix = if isempty(groups)
        similar(x, length(x), 0)
    else
        stack(_ -> vec(similar(x)), groups; dims = 2)
    end
    batched_seed_prep = ntuple(b -> copy(seed_prep), Val(B))
    batched_seeds = [
        ntuple(b -> seeds[1 + ((a - 1) * B + (b - 1)) % N], Val(B)) for a in 1:A
    ]
    batched_results = [ntuple(b -> similar(x), Val(B)) for _ in batched_seeds]
    hvp_prep = DI.prepare_hvp_nokwarg(
        strict, f, dense_backend, x, batched_seed_prep, contexts...
    )
    degenerate_results = isempty(groups) ? ntuple(b -> similar(x), Val(B)) : nothing
    return SMCSparseHessianPrep(
        _sig,
        batch_size_settings,
        sparsity,
        coloring_result,
        compressed_matrix,
        batched_seed_prep,
        batched_seeds,
        batched_results,
        hvp_prep,
        degenerate_results,
    )
end

function _sparse_hessian_aux!(
        f::F,
        grad,
        hess,
        prep::SMCSparseHessianPrep{SIG, <:DI.BatchSizeSettings{B}},
        backend::AutoSparse,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {F, SIG, B, C}
    (;
        batch_size_settings,
        coloring_result,
        compressed_matrix,
        batched_seed_prep,
        batched_seeds,
        batched_results,
        hvp_prep,
    ) = prep
    (; N) = batch_size_settings
    dense_backend = dense_ad(backend)

    hvp_prep_same = DI.prepare_hvp_same_point(
        f, hvp_prep, dense_backend, x, batched_seed_prep, contexts...
    )

    for a in eachindex(batched_seeds, batched_results)
        hvp_args = (batched_results[a], hvp_prep_same, dense_backend, x, batched_seeds[a])
        if !isnothing(grad) && a == firstindex(batched_seeds)
            DI.gradient_and_hvp!(f, grad, hvp_args..., contexts...)
        else
            DI.hvp!(f, hvp_args..., contexts...)
        end

        for b in eachindex(batched_results[a])
            copyto!(
                view(compressed_matrix, :, 1 + ((a - 1) * B + (b - 1)) % N),
                vec(batched_results[a][b]),
            )
        end
    end

    if !isnothing(grad) && !isnothing(prep.degenerate_results)
        # Degenerate all-zero sparsity pattern: the loop above never ran, so
        # extract the gradient with a throwaway HVP on the preparation seeds
        # (a first-order gradient prep would not work for HVP-only backends).
        DI.gradient_and_hvp!(
            f,
            grad,
            prep.degenerate_results,
            hvp_prep_same,
            dense_backend,
            x,
            batched_seed_prep,
            contexts...,
        )
    end

    decompress!(hess, compressed_matrix, coloring_result)
    return hess
end

function DI.hessian!(
        f::F,
        hess,
        prep::SMCSparseHessianPrep,
        backend::AutoSparse,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f, prep, backend, x, contexts...)
    return _sparse_hessian_aux!(f, nothing, hess, prep, backend, x, contexts...)
end

function DI.hessian(
        f::F, prep::SMCSparseHessianPrep, backend::AutoSparse, x, contexts::Vararg{DI.Context, C}
    ) where {F, C}
    DI.check_prep(f, prep, backend, x, contexts...)
    hess = similar(sparsity_pattern(prep), eltype(x))
    return DI.hessian!(f, hess, prep, backend, x, contexts...)
end

function DI.value_gradient_and_hessian!(
        f::F,
        grad,
        hess,
        prep::SMCSparseHessianPrep,
        backend::AutoSparse,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {F, C}
    DI.check_prep(f, prep, backend, x, contexts...)
    # there is no fused value_gradient_and_hvp operator, so the primal value
    # costs one extra call to f here
    y = f(x, map(DI.unwrap, contexts)...)
    _sparse_hessian_aux!(f, grad, hess, prep, backend, x, contexts...)
    return y, grad, hess
end

function DI.value_gradient_and_hessian(
        f::F, prep::SMCSparseHessianPrep, backend::AutoSparse, x, contexts::Vararg{DI.Context, C}
    ) where {F, C}
    DI.check_prep(f, prep, backend, x, contexts...)
    grad_buffer = similar(x)
    hess = similar(sparsity_pattern(prep), eltype(x))
    y, _, _ = DI.value_gradient_and_hessian!(
        f, grad_buffer, hess, prep, backend, x, contexts...
    )
    grad = if DI.ismutable_array(x)
        grad_buffer
    else
        # the fused path needs a mutable buffer, but the gradient returned to
        # the caller should live in the same vector space as x (e.g. SArray)
        map(+, zero(x), grad_buffer)
    end
    return y, grad, hess
end
