#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

enum class Layout {
  Other = 0,
  RowMajor = 1,
  ColumnMajor = 2,
};

constexpr std::size_t kDynamicExtent = size_t(-1);

struct Dimension {
  static constexpr Dimension dynamic() { return Dimension(); }

  constexpr Dimension withShape(std::size_t newShape) const {
    return Dimension{newShape, stride};
  }

  std::size_t shape{kDynamicExtent};
  std::size_t stride{kDynamicExtent};
};

namespace detail {
template <std::size_t kExtent> class ExtentHolder {
public:
  explicit ExtentHolder(std::size_t extent) {
    static_assert(kExtent != kDynamicExtent);
    if (extent != kExtent) {
      throw std::runtime_error("Extent mismatch");
    }
  }
  std::size_t get() const { return kExtent; }
};

template <> class ExtentHolder<kDynamicExtent> {
public:
  explicit ExtentHolder(std::size_t extent) : extent_(extent) {
    assert(extent != kDynamicExtent);
  }
  std::size_t get() const { return extent_; }

private:
  std::size_t extent_;
};

template <Dimension kDim>
class DimensionHolder
    : std::tuple<ExtentHolder<kDim.shape>, ExtentHolder<kDim.stride>> {
private:
  using Base = std::tuple<ExtentHolder<kDim.shape>, ExtentHolder<kDim.stride>>;

  Base const &base() const { return static_cast<Base const &>(*this); }

  static constexpr bool isConvertible(Dimension from, Dimension to) {
    auto isConvertible = [](std::size_t from, std::size_t to) {
      return from == kDynamicExtent || to == kDynamicExtent || from == to;
    };
    return isConvertible(from.shape, to.shape) &&
           isConvertible(from.stride, to.stride);
  }

  static constexpr bool isExplicit(Dimension from, Dimension to) {
    auto isExplicit = [](std::size_t from, std::size_t to) {
      return from == kDynamicExtent && to != kDynamicExtent;
    };
    return isExplicit(from.shape, to.shape) &&
           isExplicit(from.stride, to.stride);
  }

public:
  constexpr DimensionHolder(Dimension const &dim)
      : Base(dim.shape, dim.stride) {
    // Already validated by Extent constructor
    assert(kDim.shape == kDynamicExtent || kDim.shape == dim.shape);
    assert(kDim.stride == kDynamicExtent || kDim.stride == dim.stride);
  }

  template <Dimension kOtherDim,
            typename Enable = std::enable_if_t<isConvertible(kOtherDim, kDim)>>
  constexpr explicit(isExplicit(kOtherDim, kDim))
      DimensionHolder(DimensionHolder<kOtherDim> const &other)
      : DimensionHolder(other.dimension()) {}

  constexpr size_t shape() const { return std::get<0>(base()).get(); }
  constexpr size_t stride() const { return std::get<1>(base()).get(); }
  constexpr Dimension dimension() const { return {shape(), stride()}; }
};
} // namespace detail

template <typename T, Dimension... kDimensions>
class View : private std::tuple<detail::DimensionHolder<kDimensions>...> {
private:
  using Base = std::tuple<detail::DimensionHolder<kDimensions>...>;

  Base const &base() const { return static_cast<Base const &>(*this); }

  template <std::size_t kIndex> constexpr std::size_t getShapeAtIndex() const {
    return std::get<kIndex>(base()).shape();
  }

  template <std::size_t kIndex> constexpr std::size_t getStrideAtIndex() const {
    return std::get<kIndex>(base()).stride();
  }

  template <std::size_t kIndex>
  constexpr Dimension getDimensionAtIndex() const {
    return std::get<kIndex>(base()).dimension();
  }

  template <Dimension From, typename To> using ToType = To;

  template <typename U, ToType<kDimensions, Dimension>... kOtherDimensions>
  static constexpr bool isConvertible() {
    if (!std::is_convertible_v<U, T>) {
      return false;
    }
    auto convertible = [](Dimension from, Dimension to) {
      auto convertible = [](std::size_t from, std::size_t to) {
        return from == kDynamicExtent || to == kDynamicExtent || from == to;
      };
      return convertible(from.shape, to.shape) &&
             convertible(from.stride, to.stride);
    };
    return (convertible(kOtherDimensions, kDimensions) && ...);
  }

  template <typename U, ToType<kDimensions, Dimension>... kOtherDimensions>
  static constexpr bool isExplicit() {
    auto isExplicit = [](Dimension from, Dimension to) {
      auto isExplicit = [](std::size_t from, std::size_t to) {
        return from == kDynamicExtent && to != kDynamicExtent;
      };
      return isExplicit(from.shape, to.shape) &&
             isExplicit(from.stride, to.stride);
    };
    return (isExplicit(kOtherDimensions, kDimensions) && ...);
  }

  static constexpr std::size_t kNDims = sizeof...(kDimensions);

  template <typename Fn, std::size_t... kIdxs>
  void enumerateDimensions(std::index_sequence<kIdxs...>, Fn &&fn) const {
    (fn(kIdxs, std::get<kIdxs>(base())), ...);
  }

  template <typename U, Dimension... kOtherDimensions> friend class View;

  template <std::size_t... kIndices>
  Base construct(std::index_sequence<kIndices...>,
                 std::array<std::size_t, kNDims> const &shapes,
                 std::array<std::size_t, kNDims> const &strides) {
    return Base{{shapes[kIndices], strides[kIndices]}...};
  }

public:
  constexpr std::size_t ndims() const { return kNDims; }

  constexpr View(T *data, std::array<std::size_t, kNDims> const &shapes,
                 std::array<std::size_t, kNDims> const &strides)
      : Base(construct(std::make_index_sequence<kNDims>{}, shapes, strides)),
        data_(data) {}

  constexpr View(T *data, ToType<kDimensions, Dimension>... dims)
      : Base(dims...), data_(data) {}

  template <typename U, ToType<kDimensions, Dimension>... kOtherDimensions,
            typename Enable =
                std::enable_if_t<isConvertible<U, kOtherDimensions...>()>>
  explicit(isExplicit<U, kOtherDimensions...>())
      View(View<U, kOtherDimensions...> const &other)
      : Base((Base const &)other), data_(other.data_) {}

  constexpr std::array<std::size_t, kNDims> shapes() const {
    std::array<std::size_t, kNDims> result;
    enumerateDimensions(std::make_index_sequence<kNDims>{},
                        [&result](std::size_t idx, auto const &dim) {
                          result[idx] = dim.shape();
                        });
    return result;
  }

  constexpr std::array<std::size_t, kNDims> strides() const {
    std::array<std::size_t, kNDims> result;
    enumerateDimensions(std::make_index_sequence<kNDims>{},
                        [&result](std::size_t idx, auto const &dim) {
                          result[idx] = dim.stride();
                        });
    return result;
  }

  constexpr std::size_t shape(std::size_t idx) const { return shapes()[idx]; }

  constexpr std::size_t stride(std::size_t idx) const { return stride()[idx]; }

  constexpr T *data() const { return data_; }

  std::size_t offset(std::array<std::size_t, kNDims> const &indices) const {
    auto const strides = this->strides();
    std::size_t offset = 0;
    for (std::size_t i = 0; i < kNDims; ++i) {
      assert(indices[i] < shape(i));
      offset += indices[i] * strides[i];
    }
    return offset;
  }

  std::size_t offset(ToType<kDimensions, std::size_t>... indices) const {
    return offset({indices...});
  }

  constexpr T &at(ToType<kDimensions, std::size_t>... indices) const {
    return data()[offset(indices...)];
  }

  constexpr T &at(std::array<std::size_t, kNDims> indices) const {
    return data()[offset(indices)];
  }

  constexpr T *ptr(ToType<kDimensions, std::size_t>... indices) const {
    return data() + offset(indices...);
  }

  constexpr T *ptr(std::array<std::size_t, kNDims> indices) const {
    return data() + offset(indices);
  }

private:
  template <std::size_t... kIndices, std::size_t... kShapes>
  constexpr auto
  sliceImpl(std::array<std::size_t, kNDims> const &begin,
            std::array<std::size_t, kNDims> const &end,
            std::index_sequence<kIndices...>,
            std::integer_sequence<std::size_t, kShapes...>) const {
#ifndef NDEBUG
    for (std::size_t i = 0; i < begin.size(); ++i) {
      assert(begin[i] <= end[i]);
    }
#endif
    return View<T, kDimensions.withShape(kShapes)...>(
        &at(begin), (getDimensionAtIndex<kIndices>().withShape(
                        end[kIndices] - begin[kIndices]))...);
  }

public:
  template <std::size_t... kShapes,
            typename Enable = std::enable_if_t<sizeof...(kShapes) == 0 ||
                                               sizeof...(kShapes) == kNDims>>
  constexpr auto slice(std::array<std::size_t, kNDims> const &begin,
                       std::array<std::size_t, kNDims> const &end) const {
    if constexpr (sizeof...(kShapes) == 0) {
      constexpr auto toDynamic = [](auto) { return kDynamicExtent; };
      return sliceImpl(
          begin, end, std::make_index_sequence<kNDims>{},
          std::integer_sequence<std::size_t, toDynamic(kDimensions)...>{});
    } else {
      return sliceImpl(begin, end, std::make_index_sequence<kNDims>{},
                       std::integer_sequence<std::size_t, kShapes...>{});
    }
  }

  bool isContiguousRowMajor() const {
    auto const shapes = this->shapes();
    auto const strides = this->strides();
    if (shapes[kNDims - 1] != 1) {
      return false;
    }
    for (std::size_t i = 0; i < kNDims - 1; ++i) {
      if (strides[i] != shapes[i + 1] * strides[i + 1]) {
        return false;
      }
    }
    return true;
  }

  std::size_t numElements() const {
    auto const shapes = this->shapes();
    std::size_t result = 1;
    for (std::size_t i = 0; i < shapes.size(); ++i) {
      result *= shapes[i];
    }
    return result;
  }

  std::size_t numBytes() const { return numElements() * sizeof(T); }

private:
  T *data_;
};

namespace view {
namespace detail {
template <std::size_t kNDims, bool kHandleDynamic> class RowMajor {
public:
  template <typename... Ints>
  constexpr RowMajor(Ints... shapes) : shapes_{shapes...} {}

  constexpr RowMajor(std::array<std::size_t, kNDims> const &shapes)
      : shapes_{shapes} {}

  constexpr Dimension operator()(std::size_t index) const {
    std::size_t shape = shapes_[index];
    std::size_t stride = 1;
    for (std::size_t i = index; i-- > 0;) {
      if constexpr (kHandleDynamic) {
        if (shapes_[i] == kDynamicExtent) {
          stride = kDynamicExtent;
          break;
        }
      }
      stride *= shapes_[i];
    }
    return Dimension{shape, stride};
  }

private:
  std::array<std::size_t, kNDims> shapes_;
};

template <typename T, std::size_t... kShapes, typename... Ints,
          std::size_t... kIndices>
auto rowMajor(std::index_sequence<kIndices...>,
              std::integer_sequence<std::size_t, kShapes...>, T *data,
              Ints... shapes) {
  static_assert(sizeof...(kShapes) == sizeof...(kIndices));
  static_assert(sizeof...(Ints) == sizeof...(kIndices));

  constexpr RowMajor<sizeof...(Ints), true> kRowMajorDims(kShapes...);
  RowMajor<sizeof...(Ints), false> rowMajorDims(shapes...);

  return View<T, kRowMajorDims(kIndices)...>(data, rowMajorDims(kIndices)...);
}

template <typename T, std::size_t... kShapes, std::size_t... kIndices>
auto rowMajor(std::index_sequence<kIndices...>,
              std::integer_sequence<std::size_t, kShapes...>, T *data,
              std::array<std::size_t, sizeof...(kIndices)> const &shapes) {
  static_assert(sizeof...(kShapes) == sizeof...(kIndices));

  constexpr RowMajor<sizeof...(kIndices), true> kRowMajorDims(kShapes...);
  RowMajor<sizeof...(kIndices), false> rowMajorDims(shapes);

  return View<T, kRowMajorDims(kIndices)...>(data, rowMajorDims(kIndices)...);
}

template <typename T> constexpr T constant(T _, T to) { return to; }

template <typename T, T kVal, std::size_t... kInput>
std::integer_sequence<T, constant(kInput, kVal)...>
constantSequence(std::integer_sequence<T, kInput...>) {
  return {};
}

template <typename T, T kVal, std::size_t kSize>
using ConstantSequence =
    decltype(constantSequence<T, kVal>(std::make_integer_sequence<T, kSize>{}));
} // namespace detail

template <
    typename T, std::size_t... kShapes, typename... Ints,
    typename Enable = std::enable_if_t<
        (std::is_convertible_v<Ints, std::size_t> && ...) &&
        (sizeof...(kShapes) == 0 || sizeof...(kShapes) == sizeof...(Ints))>>
auto rowMajor(T *data, Ints... shapes) {
  if constexpr (sizeof...(kShapes) == 0) {
    constexpr auto toDynamic = [](auto) { return kDynamicExtent; };
    return detail::rowMajor(
        std::make_index_sequence<sizeof...(Ints)>(),
        detail::ConstantSequence<std::size_t, kDynamicExtent,
                                 sizeof...(Ints)>{},
        data, shapes...);
  } else {
    return detail::rowMajor(std::make_index_sequence<sizeof...(Ints)>(),
                            std::integer_sequence<std::size_t, kShapes...>{},
                            data, shapes...);
  }
}

template <typename T, std::size_t... kShapes, std::size_t kNDims,
          typename Enable = std::enable_if_t<(sizeof...(kShapes) == 0 ||
                                              sizeof...(kShapes) == kNDims)>>
auto rowMajor(T *data, std::array<std::size_t, kNDims> const &shapes) {
  if constexpr (sizeof...(kShapes) == 0) {
    constexpr auto toDynamic = [](auto) { return kDynamicExtent; };
    return detail::rowMajor(
        std::make_index_sequence<kNDims>(),
        detail::ConstantSequence<std::size_t, kDynamicExtent, kNDims>{}, data,
        shapes);
  } else {
    return detail::rowMajor(std::make_index_sequence<kNDims>(),
                            std::integer_sequence<std::size_t, kShapes...>{},
                            data, shapes);
  }
}
} // namespace view

template <typename T, Dimension kDimX = Dimension::dynamic()>
using View1D = View<T, kDimX>;

template <typename T, Dimension kDimX = Dimension::dynamic(),
          Dimension kDimY = Dimension::dynamic()>
using View2D = View<T, kDimX, kDimY>;

template <typename T, Dimension kDimX = Dimension::dynamic(),
          Dimension kDimY = Dimension::dynamic(),
          Dimension kDimZ = Dimension::dynamic()>
using View3D = View<T, kDimX, kDimY, kDimZ>;