using Printf
using HTTP

function get_destination(i, create)
  parent = (
    joinpath(
      joinpath(
        dirname(dirname(Base.source_path())), "data"
      ), @sprintf("case_%05d", i)
    )
  )
  file = joinpath(parent, "imaging.nii.gz")

  if create
    mkpath(parent)
  end

  return file
end

function download(i)
  imaging_url = @sprintf("https://kits19.sfo2.digitaloceanspaces.com/master_%05d.nii.gz", i)
  tmp = tempdir()
  HTTP.download(imaging_url, tmp)
  mv(joinpath(tmp, @sprintf("master_%05d.nii.gz", i)), get_destination(i, true))
end

n_to_download = 0
left_to_download = Int64[]
for i = 0:299
  dst = get_destination(i, false)
  if !isfile(dst)
    append!(left_to_download, i)
    global n_to_download = n_to_download + 1
  end
end

println(@sprintf("%d cases to download...", n_to_download))
for i = 1:n_to_download
  println(@sprintf("%d/%d...", i, n_to_download))
  download(left_to_download[i])
end
