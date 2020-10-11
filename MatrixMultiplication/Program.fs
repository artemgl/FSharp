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
    let iterations = 10
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
            
Main "NVIDIA*" 10