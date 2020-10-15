module MatrixMultiply

open OpenCL.Net
open Brahma.OpenCL
open Brahma.FSharp.OpenCL.Core
open Microsoft.FSharp.Quotations
open Brahma.FSharp.OpenCL.Extensions

let random = new System.Random()
        
let MakeMatrix rows cols =
    Array.init (rows * cols) (fun i -> float32 (random.NextDouble()))
   
let PrintMatrix (array:float32 []) rows cols =
    for i in 0 .. rows - 1 do
        for j in 0 .. cols - 1 do
            printf "%A " array.[i * cols + j]
        printfn ""

let Main platformName mSize =    

    let m1 = (MakeMatrix mSize mSize)
    let m2 = (MakeMatrix mSize mSize)
    let localWorkSize = 2
    let deviceType = DeviceType.Default

    let provider =
        try  ComputeProvider.Create(platformName, deviceType)
        with 
        | ex -> failwith ex.Message

    let mutable commandQueue = new CommandQueue(provider, provider.Devices |> Seq.head)

    let aValues = m1
    let bValues = m2
    let cValues = Array.zeroCreate(mSize * mSize)

    printfn "Matrix A:"
    PrintMatrix aValues mSize mSize

    printfn "Matrix B:"
    PrintMatrix bValues mSize mSize

    let command = 
        <@
            fun (r:_2D) (a:array<_>) (b:array<_>) (c:array<_>) -> 
                let tx = r.GlobalID0
                let ty = r.GlobalID1
                let mutable buf = c.[ty * mSize + tx]
                for k in 0 .. mSize - 1 do
                    buf <- buf + (a.[ty * mSize + k] * b.[k * mSize + tx])
                c.[ty * mSize + tx] <- buf
        @>

    let kernel, kernelPrepare, kernelRun = provider.Compile command
    let d = new _2D(mSize, mSize, localWorkSize, localWorkSize)
    kernelPrepare d aValues bValues cValues
    
    commandQueue.Add(kernelRun()).Finish() |> ignore

    let _ = commandQueue.Add(cValues.ToHost provider).Finish()

    printfn "Matrix A * B:"
    PrintMatrix cValues mSize mSize

    commandQueue.Dispose()
    provider.CloseAllBuffers()
    provider.Dispose()    

let MakeSparseMatrix rows cols =
    Array.init ((rows + cols) / 2) (fun i -> float32 (random.NextDouble()), random.Next(0, rows - 1), random.Next(0, cols - 1))

let TensorProduct (aValues:array<_*_*_>) (bValues:array<_*_*_>) bRows bCols (cValues:array<_*_*_>) =
    let mutable m = 0
    for (a, i, j) in aValues do
        for (b, k, l) in bValues do
            cValues.[m] <- (a * b, i * bRows + k, j * bCols + l)
            m <- m + 1

let SparseMatrixMultiply platformName aRows aCols bRows bCols =

    let aLength = (aRows + aCols) / 2
    let bLength = (bRows + bCols) / 2
    let cLength = aLength * bLength

    let aValues = MakeSparseMatrix aRows aCols
    let bValues = MakeSparseMatrix bRows bCols
    let cNormal = Array.init cLength (fun i -> float32 0.0, 0, 0)
    let iterations = 100
    let localWorkSize = 2
    let deviceType = DeviceType.Default

    printfn "Tensor product of matrix %Ax%A and matrix %Ax%A %A times using .NET..." aRows aCols bRows bCols iterations
    let cpuStart = System.DateTime.Now
    for i in 0 .. iterations - 1 do
        TensorProduct aValues bValues bRows bCols cNormal
    let cpuTime = System.DateTime.Now - cpuStart
    printfn "done."
    
    let provider =
        try ComputeProvider.Create(platformName, deviceType)
        with 
        | ex -> failwith ex.Message
    
    let mutable commandQueue = new CommandQueue(provider, provider.Devices |> Seq.head)

    let command = 
        <@
            fun (rng:_2D) (aVals:array<_>) (aRow:array<_>) (aCol:array<_>)
                (bVals:array<_>) (bRow:array<_>) (bCol:array<_>) (cVals:array<_>) (cRows:array<_>) (cCols:array<_>) ->
                    let x = rng.GlobalID0
                    let y = rng.GlobalID1
                    let index = x * bLength + y
                    let (aElement, i, j) = aVals.[x], aRow.[x], aCol.[x]
                    let (bElement, k, l) = bVals.[y], bRow.[y], bCol.[y]
                    cVals.[index] <- aElement * bElement
                    cRows.[index] <- i * bRows + k
                    cCols.[index] <- j * bCols + l
        @>

    let cParallel = Array.init cLength (fun i -> float32 0.0, 0, 0)

    let kernel, kernelPrepare, kernelRun = provider.Compile command
    let d = new _2D(aLength, bLength, localWorkSize, localWorkSize)
    let aVal, aRow, aCol = Array.unzip3 aValues
    let bVal, bRow, bCol = Array.unzip3 bValues
    let cParVals, cParRows, cParCols = Array.unzip3 cParallel
    kernelPrepare d aVal aRow aCol bVal bRow bCol cParVals cParRows cParCols

    printfn "Tensor product of matrix %Ax%A and matrix %Ax%A %A times using OpenCL and selected platform/device : %A ..." aRows aCols bRows bCols iterations provider
    let gpuStart = System.DateTime.Now
    for i in 0 .. iterations - 1 do
        commandQueue.Add(kernelRun()).Finish() |> ignore
    let gpuTime = System.DateTime.Now - gpuStart
    printfn "done."

    let _ = commandQueue.Add(cParVals.ToHost provider).Finish()
    let _ = commandQueue.Add(cParRows.ToHost provider).Finish()
    let _ = commandQueue.Add(cParCols.ToHost provider).Finish()
    for i in 0 .. cLength - 1 do
        cParallel.[i] <- cParVals.[i], cParRows.[i], cParCols.[i]

    printfn "Verifying results..."
    cNormal |> Array.sortInPlaceBy (fun (_, i, j) -> i, j) |> ignore
    cParallel |> Array.sortInPlaceBy (fun (_, i, j) -> i, j) |> ignore
    let mutable isSuccess = true
    for i in 0 .. cLength - 1 do
        let (valNorm, rowNorm, colNorm) = cNormal.[i]
        let (valPar, rowPar, colPar) = cParallel.[i]
        if isSuccess && not (System.Math.Abs(float32 (valNorm - valPar)) <= 0.01f && rowNorm = rowPar && colNorm = colPar)
        then
            isSuccess <- false
            printfn "Expected: %A Actual: %A Error: %A" valNorm valPar (System.Math.Abs(float32 (valNorm - valPar)))
    printfn "done."

    cpuTime.TotalMilliseconds / float iterations |> printfn "Avg. time, F#: %A"
    gpuTime.TotalMilliseconds / float iterations |> printfn "Avg. time, OpenCL: %A"

    commandQueue.Dispose()
    provider.CloseAllBuffers()
    provider.Dispose()

SparseMatrixMultiply "NVIDIA*" 200 200 200 200