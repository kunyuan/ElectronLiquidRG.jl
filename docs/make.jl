using ElectronLiquidRG
using Documenter

DocMeta.setdocmeta!(ElectronLiquidRG, :DocTestSetup, :(using ElectronLiquidRG); recursive=true)

makedocs(;
    modules=[ElectronLiquidRG],
    authors="Kun Chen",
    repo="https://github.com/kunyuan/ElectronLiquidRG.jl/blob/{commit}{path}#{line}",
    sitename="ElectronLiquidRG.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://kunyuan.github.io/ElectronLiquidRG.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kunyuan/ElectronLiquidRG.jl",
    devbranch="main",
)
